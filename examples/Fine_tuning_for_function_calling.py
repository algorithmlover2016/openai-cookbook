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
    if not tools:
        params.pop("tools")
        params.pop("tool_choice")

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

placeholder_int = 'fill_in_int'
placeholder_string = 'fill_in_string'

def generate_permutations(params: Dict[str, Dict[str, Any]]) -> Generator[Dict[str, Any], None, None]:
    """
    Generates all possible permutations for given parameters.

    :param params: Parameter dictionary containing required and optional fields.
    :return: A generator yielding each permutation.
    """

    # Extract the required fields from the parameters
    required_fields = params.get('required', [])

    # Generate permutations for required fields
    required_permutations = generate_required_permutations(params, required_fields)

    # Generate optional permutations based on each required permutation
    for required_perm in required_permutations:
        yield from generate_optional_permutations(params, required_perm)


def generate_required_permutations(params: Dict[str, Dict[str, Any]], required_fields: List[str]) -> List[Dict[str, Any]]:
    """
    Generates permutations for the required fields.

    :param params: Parameter dictionary.
    :param required_fields: List of required fields.
    :return: A list of permutations for required fields.
    """

    # Get all possible values for each required field
    required_values = [get_possible_values(params, field) for field in required_fields]

    # Generate permutations from possible values
    return [dict(zip(required_fields, values)) for values in itertools.product(*required_values)]


def generate_optional_permutations(params: Dict[str, Dict[str, Any]], base_perm: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
    """
    Generates permutations for optional fields based on a base permutation.

    :param params: Parameter dictionary.
    :param base_perm: Base permutation dictionary.
    :return: A generator yielding each permutation for optional fields.
    """

    # Determine the fields that are optional by subtracting the base permutation's fields from all properties
    optional_fields = set(params['properties']) - set(base_perm)

    # Iterate through all combinations of optional fields
    for field_subset in itertools.chain.from_iterable(itertools.combinations(optional_fields, r) for r in range(len(optional_fields) + 1)):

        # Generate product of possible values for the current subset of fields
        for values in itertools.product(*(get_possible_values(params, field) for field in field_subset)):

            # Create a new permutation by combining base permutation and current field values
            new_perm = {**base_perm, **dict(zip(field_subset, values))}

            yield new_perm


def get_possible_values(params: Dict[str, Dict[str, Any]], field: str) -> List[Any]:
    """
    Retrieves possible values for a given field.

    :param params: Parameter dictionary.
    :param field: The field for which to get possible values.
    :return: A list of possible values.
    """

    # Extract field information from the parameters
    field_info = params['properties'][field]

    # Based on the field's type or presence of 'enum', determine and return the possible values
    if 'enum' in field_info:
        return field_info['enum']
    elif field_info['type'] == 'integer':
        return [placeholder_int]
    elif field_info['type'] == 'string':
        return [placeholder_string]
    elif field_info['type'] == 'boolean':
        return [True, False]
    elif field_info['type'] == 'array' and 'enum' in field_info['items']:
        enum_values = field_info['items']['enum']
        all_combinations = [list(combo) for i in range(1, len(enum_values) + 1) for combo in itertools.combinations(enum_values, i)]
        return all_combinations
    return []

INVOCATION_FILLER_PROMPT = """
1) Input reasonable values for "fill_in_string" and "fill_in_int" in the invocation here: {invocation}. Reasonable values are determined by the function definition. Use the
the entire function provided here :{function} to get context over what proper fill_in_string and fill_in_int values would be.
Example:

Input: invocation: {{
    "name": "control_camera",
    "arguments": {{
      "mode":"video",
      "duration":"fill_in_int"
    }}
}},
function:{function}

Output: invocation: {{
    "name": "control_camera",
    "arguments": {{
      "mode":"video",
      "duration": 30
    }}
}}


MAKE SURE output is just a dictionary with keys "name" and "arguments", no other text or response. You should use " instead of ' to mark string.

Input: {invocation}
Output:
"""


COMMAND_GENERATION_PROMPT= """
You are to output 2 commands, questions or statements that would generate the inputted function and parameters.
Please make the commands or questions natural, as a person would ask, and the command or questions should be varied and not repetitive.
It should not always mirror the exact technical terminology used in the function and parameters, rather reflect a conversational and intuitive request.
For instance, the prompt should not be 'turn on the dome light', as that is too technical, but rather 'turn on the inside lights'.
Another example, is the prompt should not be 'turn on the HVAC', but rather 'turn on the air conditioning'. Use language a normal driver would use, even if
it is technically incorrect but colloquially used.

RULES: ALWAYS put a backwards slash before an apostrophe or single quote '. For example, do not say don't but say don\'t.
Prompts MUST be in double quotes as well.

Example

Input: {{'name': 'calibrate_sensors','arguments': {{}}'' }}
Prompt: ["The sensors are out of whack, can you reset them", "The calibration of the drone is off, fix it please!"]

Input: {{'name': 'set_autopilot','arguments': {{'status': 'off'}}}}
Prompt: ["OK, I want to take back pilot control now","Turn off the automatic pilot I'm ready control it"]

Input: {invocation}
Prompt:
"""

# pdb.set_trace()

input_objects = []
all_but_reject = [f for f in function_list if f.get("function").get('name') != 'reject_request']

for function in all_but_reject:
    func_name = function['function']['name']
    params = function['function']['parameters']
    for arguments in generate_permutations(params):
      if any(val in arguments.values() for val in ['fill_in_int', 'fill_in_str']):
          input_object = {
              "name": func_name,
              "arguments": arguments
          }
          messages = [{"role": "user", "content": INVOCATION_FILLER_PROMPT.format(invocation=input_object,function=function)}]
          input_object = get_chat_completion(model=GPT_MODEL, messages=messages, max_tokens = 200, temperature=.1).content
          if isinstance(input_object, str):
              input_object = json.loads(input_object)
      else:
          input_object = {
              "name": func_name,
              "arguments": arguments
          }

      input_objects.append(input_object)

logging.info(json.dumps(input_objects))