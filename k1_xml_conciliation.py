from pdf2image import convert_from_path, convert_from_bytes
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError  
)
from io import BytesIO  
from dotenv import load_dotenv, find_dotenv
from typing import Any, Dict, List, Optional, Tuple, Union
import pickle
import base64
import xml.etree.ElementTree as ET
import os
from PIL import Image
import requests
from ast import literal_eval
from time import sleep
import pandas as pd


# Poppler path

poppler_path = "c:/Users/FE865KL/Downloads/poppler-windows/poppler-windows-Release-24.07.0-0/poppler-24.07.0/Library/bin" # FIXME


# LLM variables

load_dotenv(find_dotenv())

llm_config = {
    # "model": os.environ.get("AZURE_OPENAI_MODEL"),
    "model": "gpt-4", # NOTE 
    "api_type": os.environ.get("AZURE_OPENAI_API_TYPE"),
    "api_key": os.environ.get("AZURE_OPENAI_API_KEY"),
    "base_url": os.environ.get("AZURE_OPENAI_API_BASE"),
    "api_version": os.environ.get("AZURE_OPENAI_API_VERSION")
    }

COMPLETE_ENDPOINT = f'{llm_config["base_url"]}openai/deployments/{llm_config["model"]}/chat/completions?api-version={llm_config["api_version"]}'

# PDF and XML paths

# input_path = "../../../tax-gen-ai/services/plugins/py-xml-extract/tests/files"
input_path = "../tax-gen-ai/services/plugins/py-xml-extract/tests/files" # FIXME
# filename = "sample_k1.pdf" # FIXME
filename = "sample_k1_printed.pdf" # FIXME
# filename = "MN- TESTING SAMPLE RETURN.pdf" # FIXME
filename_xml = "sample_k1.xml"

xml_file = f'{input_path}/{filename_xml}'


# PDF bounding boxes
# Note: so far we are handcrafting the bounding boxes based on a specific K1 form.

# Original Sample:
bouding_boxes = [ # FIXME: sample K1
    (0,0,865,705),# H_P1
    (0,690,865,250), # P2_E_F
    (0,920,865,790), # P2_G_K
    (0,1700,865,500), # P2_L_M_N
    (860,0,865,535), # P3_1_16
    (860,530,865,1600), # P3_4c_
]

# # MN- TESTING SAMPLE RETURN.pdf
# bouding_boxes = [
#     (0,280,1697,648), # Header
#     (0,660,1697,833), # Checks
#     (0,900,1697,1460), # Questions 1 - 7
#     (0,1468,1697,2040), # Questions 7 - 13
# ]



# PDF parsing Prompt:


# --------- Misc Funcitons

# Encode to base64:
def encode_from_pil(im):
    output = BytesIO()
    im.save(output, format='JPEG')
    im_data = output.getvalue()
    # Encode
    image_data = base64.b64encode(im_data)
    if not isinstance(image_data, str):
        image_data = image_data.decode()
    return image_data

def crop(bg_removed_image_path, bounding_box):#, output_dir, image_name):
    try:
        # Load the image with background removed
        
        image = bg_removed_image_path# image = Image.open(bg_removed_image_path).convert('RGBA')

        # Extract bounding box coordinates
        x, y, w, h = bounding_box

        # Crop the image based on the bounding box
        cropped_image = image.crop((x, y, x + w, y + h))

        # # Save the final image as PNG with maximum quality settings
        # final_img_path = os.path.join(output_dir, 'crop', f'{x}_{y}_{w}_{h}.png')
        # cropped_image.save(final_img_path, 'PNG', quality=95)  # Although 'quality' has no effect on PNGs, provided for completeness
 
        return cropped_image
    
    except Exception as e:
        print(f"Error in cropping and resizing: {e}")
        return None


def extract_list(response):
    ini_marker = "```python\n"
    end_marker = "\n```"
    ini = response.find(ini_marker)
    list_str = response[ini+len(ini_marker):]
    end = list_str.find(end_marker)
    list_str = list_str[:end]
    list_str = literal_eval(list_str)
    return list_str

def extract_json_list(response):
    ini_marker = "```json\n"
    end_marker = "\n```"
    ini = response.find(ini_marker)
    list_str = response[ini+len(ini_marker):]
    end = list_str.find(end_marker)
    list_str = list_str[:end]
    list_str = literal_eval(list_str)
    return list_str

def pdf_parser(
        input_path: str = input_path,
        filename: str = filename,
        COMPLETION_ENDPOINT: str = COMPLETE_ENDPOINT,
        poppler_path: str = poppler_path,
        bounding_boxes: List[Tuple[int, int, int, int]] = bouding_boxes
        ) -> List[Dict[str, str]]:

    # Convert PDF to images

    images = convert_from_path(
        f'{input_path}/{filename}',
        poppler_path=poppler_path,
        grayscale=True,
        use_cropbox=True,)

    # Process cropped images
    k1_fields = []
    for bounding_box in bounding_boxes:
        base64_image_crop = encode_from_pil(crop(images[0], bounding_box))
        # # Configuration
        # IMAGE_PATH = "../data/block-diagrama.png"
        # encoded_image = base64.b64encode(open(IMAGE_PATH, 'rb').read()).decode('ascii')
        # encoded_image = base64_image
        headers = {
            "Content-Type": "application/json",
            "api-key": llm_config["api_key"],
        }

        # Payload for the request
        payload = {
        "messages": [
            {
            "role": "system",
            "content": [
                {
                "type": "text",
                "text": "You are an AI assistant that helps extracting and parsing information from K1 tax forms from IRS."
                }
            ]
            },
            {
            "role": "user",
            "content": [
                # {
                # "type": "text",
                # "text": """
                # You are provided with the image of a section of a k1 tax form.
                # I need you to enumerate as key:value pairs all the expected fields or checkboxes (keys), and the values.
                # If no values are present, return "--NONE--" as the value for that key.
                # For checkboxes, return "Box Checked" if checked, otherwise return "Box Unchecked". Pay special attention to the checkboxes; checkbox with an "X" in it must be considered as "Box Checked".
                # Return the information as a python list of dictionaries.
                # """
                # },
                {
                "type": "text",
                "text": """
                You are provided with the image of a section of an IRS tax form.
                I need you to extract all the expected fields or checkboxes (keys), and their respective values.

                Please consider the following instructions:
                    - provide the label of the field
                    - provide the value of the field if any. If no values are present, return "--NONE--" as the value for that key.
                        - For checkboxes, return "Box Checked" if checked, otherwise return "Box Unchecked". Pay special attention to the checkboxes; checkbox with an "X" in it must be considered as "Box Checked".
                    - Specify the field type whenever possible (e.g. %, $, checkbox, etc.)
                    - If the document is composed by different parts, specify the part number and description.
                    - Provide the information as a python list of dictionaries, with one dictionary per field.
                        - Provide the list inside the markers ```python\n and \n```
                        - Make plain dictionaries (do not put fields inside parts), e.g. field keys label, type, part_number, part_description.
                """
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image_crop}"
                }
                }
            ]
            }
        ],
        "temperature": 0.3,
        "top_p": 0.95,
        "max_tokens": 800
        }


        # Send request
        try:
            response = requests.post(COMPLETE_ENDPOINT, headers=headers, json=payload)
            response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
        except requests.RequestException as e:
            raise SystemExit(f"Failed to make the request. Error: {e}")
            break

        # Handle the response as needed (e.g., print or process)
        # print(response.json())
        k1_fields += extract_list(response.json()["choices"][-1]["message"]["content"])
        sleep(30)
    k1_fields    

    return k1_fields


# --------- XML processing

def traverse_and_print_leaves(element, level=0, parent_tags=[], k1_fields=[]):
    # Check if the element has children
    if len(element):
        # If the element has children, recursively traverse each child
        for child in element:
            tag = element.tag
            tag = tag.replace("{http://www.irs.gov/efile}", "")
            traverse_and_print_leaves(child, level + 1, parent_tags + [tag], k1_fields)
    else:
        # If the element has no children, it is a leaf node
        tag = element.tag
        tag = tag.replace("{http://www.irs.gov/efile}", "")
        parent_tags = "/".join(parent_tags)
        full_tag = f"{parent_tags}/{tag}"
        # print(f"{full_tag}")
        k1_fields.append((full_tag, element.text))

def xml_pdf_fields_conciliation(xml_fields,
                                pdf_fields,
                                ):

    reconciliation_prompt = f"""
I have two input lists, source and target. I need to match the elements from the source to the target.
The source is provided as a list of dicts with key:value pairs and the target as a list of tuples with key:value pairs. You can use key and value information, but i need to match ONLY keys (not values) from source. 
Please give me the output as a python list of tuples (x,y) where each element has the source (x) and target keys (y).
If you cant find a match for a target key, return "--NONE--" as the source key.

Provide only the requested python list of tuples as output and NOTHING ELSE. DO NOT ADD ANY ADDITIONAL TEXT OR COMMENTS.
Provide the output list with the following structure:
```python
[
(source_key1, target_key1),
(source_key2, target_key2),
 ...
]
```

Target
------

{xml_fields}

Source
------

{pdf_fields}
"""

    headers = {
        "Content-Type": "application/json",
        "api-key": llm_config["api_key"],
    }

    # Payload for the request
    payload = {
    "messages": [
        {
        "role": "system",
        "content": [
            {
            "type": "text",
            "text": "You are an AI assistant that helps extracting and parsing information from K1 tax forms from IRS."
            }
        ]
        },
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": reconciliation_prompt
            }
        ]
        }
    ],
    "temperature": 0.3,
    "top_p": 0.95,
    # "max_tokens": 800
    }


    # Send request
    try:
        response = requests.post(COMPLETE_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
    except requests.RequestException as e:
        raise SystemExit(f"Failed to make the request. Error: {e}")

    # Handle the response as needed (e.g., print or process)
    # print(response.json())

    return response.json()["choices"][0]["message"]["content"]


# --------- Main Script

def main():

    # PDF processing

    try:
        # load the list of dictionaries from the pickle file
        with open(input_path+'/k1_fields_pdf_v2.pkl', 'rb') as file:
            k1_fields = pickle.load(file)
    except:
        k1_fields = pdf_parser()
        # save the list of dictionaries to a pickle file
        with open(input_path+'/k1_fields_pdf_v2.pkl', 'wb') as file:
            pickle.dump(k1_fields, file)

    # Read XML file # TODO

    # with open(xml_file, 'r') as file:
    #     xml_content = file.read()

    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Traverse the XML tree and print all leaf tags
    xml_k1_fields = []
    traverse_and_print_leaves(root, k1_fields=xml_k1_fields)
    with open(input_path+'/k1_fields_xml.pkl', 'wb') as file:
        pickle.dump(xml_k1_fields, file)

    # Reconciliation
    response = xml_pdf_fields_conciliation(xml_k1_fields, k1_fields)
    mapping = extract_list(response)
    # for i in mapping:
    #     print(f"{i[0]}\t{i[1]}")

    # Save results:


    k1_pdf = [(k,v) for d in k1_fields for k,v in d.items()]
    df_pdf = pd.DataFrame(k1_pdf, columns=["PDF field","PDF value"])
    df_pdf.to_csv(input_path+'/k1_pdf_kv.csv', index=False, sep="\t")

    dfmap = pd.DataFrame(mapping, columns=["PDF","XML"])
    dfmap.replace("--NONE--", None)
    dfmap.to_csv(input_path+'/k1_XML_fields_mapping.csv', index=False, sep="\t")

    # create a new datafrme from df_pdf and dfmap joining on "PDF field" with "PDF"
    # df = df_pdf.merge(dfmap, left_on="PDF field", right_on="PDF", how="left")
    # df.to_csv(input_path+'/k1_pdf_xml_conciliation.csv', index=False, sep="\t")

    # outer join to get the missing fields from both sides
    df_outer = df_pdf.merge(dfmap, left_on="PDF field", right_on="PDF", how="outer")
    df_outer.to_csv(input_path+'/k1_pdf_xml_conciliation.csv', sep="\t")


if __name__ == "__main__":
    main()