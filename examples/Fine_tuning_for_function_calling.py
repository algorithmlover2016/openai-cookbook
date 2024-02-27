#-*- coding:utf-8 -*-

import ast
import itertools
import json
import logging
import numpy as np
import os
from openai import OpenAI, AzureOpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt
import traceback
from typing import Any, Dict, List, Generator

from config import azure_embedding_model, GPT_MODEL, azure_endpoint, api_key as azure_api_key, api_version as azure_api_version

GPT_MODEL = GPT_MODEL
EMBEDDING_MODEL = azure_embedding_model
# client = OpenAI()
client = AzureOpenAI(
    api_key = azure_api_key, 
    api_version = azure_api_version,
    azure_endpoint = azure_endpoint
)

@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(
    messages: list[dict[str]],
    model: str=GPT_MODEL,
    tools=None,
    tool_choice="none",
    stop=None,
    temperature=0.7,
    max_tokens=4000,
    top_p=0.95,
    frequency_penalty=0):

    params = {
        'messages': messages,
        'model': model,
        'tools': tools,
        'tool_choice': tool_choice,
        'stop': stop,
        'temperature': temperature,
        'max_tokens': max_tokens,
        'top_p': top_p,
        'frequency_penalty': frequency_penalty,
    }

    try:
        response = client.chat.completions.create(**params)
        return response
    except Exception as e:
        # Handle the exception
        exception_type = type(e).__name__
        exception_message = str(e)
        traceback_info = traceback.format_exc()

        error_msg = f"{exception_type}: {exception_message}\r\nTraceback:\r\n{traceback_info}"
        logging.error(error_msg)
        return e

@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def get_chat_completion(
    messages: list[dict[str, str]],
    model: str = GPT_MODEL,
    max_tokens=500,
    temperature=1.0,
    tools=None,
    top_p = 0.95,
    frequency_penalty=0,
    tool_choice='none'
):
    params = {
        'model': model,
        'messages': messages,
        'max_tokens': max_tokens,
        'temperature': temperature,
        'tools': tools,
        'frequency_penalty': frequency_penalty,
        'tool_choice': tool_choice,
        'top_p': top_p
    }

    try:
        completion = client.chat.completions.create(**params)
        return completion.choices[0].message
    except Exception as e:
        # Handle the exception
        exception_type = type(e).__name__
        exception_message = str(e)
        traceback_info = traceback.format_exc()

        error_msg = f"{exception_type}: {exception_message}\r\nTraceback:\r\n{traceback_info}"
        logging.error(error_msg)
        return e 


DRONE_SYSTEM_PROMPT = """You are an intelligent AI that controls a drone. Given a command or request from the user,
call one of your functions to complete the request. If the request cannot be completed by your available functions, call the reject_request function.
If the request is ambiguous or unclear, reject the request."""

function_list = [
    {
        "type": "function",
        "function": {
            "name": "takeoff_drone",
            "description": "Initiate the drone's takeoff sequence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "altitude": {
                        "type": "integer",
                        "description": "Specifies the altitude in meters to which the drone should ascend.",
                    }
                },
                "required": ["altitude"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "land_drone",
            "description": "Land the drone at its current location or a specified landing point.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "enum": ["current", "home_base", "custom"],
                        "description": "Specifies the landing location for the drone.",
                    },
                    "coordinates": {
                        "type": "object",
                        "description": "GPS coordinates for custom landing location. Required if location is 'custom'.",
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "control_drone_movement",
            "description": "Direct the drone's movement in a specific direction.",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["forward", "backward", "left", "right", "up", "down"],
                        "description": "Direction in which the drone should move.",
                    },
                    "distance": {
                        "type": "integer",
                        "description": "Distance in meters the drone should travel in the specified direction.",
                    },
                },
                "required": ["direction", "distance"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_drone_speed",
            "description": "Adjust the speed of the drone.",
            "parameters": {
                "type": "object",
                "properties": {
                    "speed": {
                        "type": "integer",
                        "description": "Specifies the speed in km/h.",
                    }
                },
                "required": ["speed"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "control_camera",
            "description": "Control the drone's camera to capture images or videos.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["photo", "video", "panorama"],
                        "description": "Camera mode to capture content.",
                    },
                    "duration": {
                        "type": "integer",
                        "description": "Duration in seconds for video capture. Required if mode is 'video'.",
                    },
                },
                "required": ["mode"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "control_gimbal",
            "description": "Adjust the drone's gimbal for camera stabilization and direction.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tilt": {
                        "type": "integer",
                        "description": "Tilt angle for the gimbal in degrees.",
                    },
                    "pan": {
                        "type": "integer",
                        "description": "Pan angle for the gimbal in degrees.",
                    },
                },
                "required": ["tilt", "pan"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_drone_lighting",
            "description": "Control the drone's lighting for visibility and signaling.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["on", "off", "blink", "sos"],
                        "description": "Lighting mode for the drone.",
                    }
                },
                "required": ["mode"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "return_to_home",
            "description": "Command the drone to return to its home or launch location.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_battery_saver_mode",
            "description": "Toggle battery saver mode.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["on", "off"],
                        "description": "Toggle battery saver mode.",
                    }
                },
                "required": ["status"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_obstacle_avoidance",
            "description": "Configure obstacle avoidance settings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["on", "off"],
                        "description": "Toggle obstacle avoidance.",
                    }
                },
                "required": ["mode"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_follow_me_mode",
            "description": "Enable or disable 'follow me' mode.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["on", "off"],
                        "description": "Toggle 'follow me' mode.",
                    }
                },
                "required": ["status"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calibrate_sensors",
            "description": "Initiate calibration sequence for drone's sensors.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_autopilot",
            "description": "Enable or disable autopilot mode.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["on", "off"],
                        "description": "Toggle autopilot mode.",
                    }
                },
                "required": ["status"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "configure_led_display",
            "description": "Configure the drone's LED display pattern and colors.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "enum": ["solid", "blink", "pulse", "rainbow"],
                        "description": "Pattern for the LED display.",
                    },
                    "color": {
                        "type": "string",
                        "enum": ["red", "blue", "green", "yellow", "white"],
                        "description": "Color for the LED display. Not required if pattern is 'rainbow'.",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_home_location",
            "description": "Set or change the home location for the drone.",
            "parameters": {
                "type": "object",
                "properties": {
                    "coordinates": {
                        "type": "object",
                        "description": "GPS coordinates for the home location.",
                    }
                },
                "required": ["coordinates"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reject_request",
            "description": "Use this function if the request is not possible.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


straightforward_prompts = ['Land the drone at the home base',
 'Take off the drone to 50 meters',
 'change speed to 15 kilometers per hour',
  'turn into an elephant!']

import pdb
pdb.set_trace()

for prompt in straightforward_prompts:
    messages = []
    messages.append({"role": "system", "content": DRONE_SYSTEM_PROMPT})
    messages.append({"role": "user", "content": prompt})
    completion = get_chat_completion(model=GPT_MODEL, messages=messages,tools=function_list, tool_choice="auto")
    logging.info(prompt)
    logging.info(completion.tool_calls[0].function)

challenging_prompts = ['Play pre-recorded audio message',
                       'Initiate live-streaming on social media',
                       'Scan environment for heat signatures',
                       'Enable stealth mode',
                       "Change drone's paint job color"]


for prompt in challenging_prompts:
    messages = []
    messages.append({"role": "system", "content": DRONE_SYSTEM_PROMPT})
    messages.append({"role": "user", "content": prompt})
    completion = get_chat_completion(model=GPT_MODEL, messages=messages,tools=function_list, tool_choice="auto")
    logging.info(prompt)
    try:
      logging.info(completion.tool_calls[0].function)
    except:
      logging.info(completion.tool_calls[0].content)
