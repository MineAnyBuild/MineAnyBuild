import os
import json
import math

def calculate_difficulty_factor(data, filename_for_log=""):
    try:
        block_amount = data.get("block_amount")
        info_3d = data.get("3d_info")

        if block_amount is None:
            print(f"  Warning ({filename_for_log}): Missing 'block_amount' key. Unable to calculate difficulty_factor.")
            return None
        if not isinstance(info_3d, dict):
            print(f"  Warning ({filename_for_log}): '3d_info' key is missing or not a dictionary. Unable to calculate difficulty_factor.")
            return None

        height = info_3d.get("height")
        width = info_3d.get("width")
        depth = info_3d.get("depth")

        required_keys_3d_info = {"height": height, "width": width, "depth": depth}
        for key, val in required_keys_3d_info.items():
            if val is None:
                print(f"  Warning ({filename_for_log}): '3d_info.{key}' key is missing. Unable to calculate difficulty_factor.")
                return None
        
        values_to_check = [block_amount, height, width, depth]
        if not all(isinstance(val, (int, float)) for val in values_to_check):
            num_values_str = f"block_amount={block_amount}, height={height}, width={width}, depth={depth}"
            print(f"  Warning ({filename_for_log}): 'block_amount' or values in '3d_info' are not numeric. Values are: {num_values_str}. Unable to calculate.")
            return None

        f_block_amount = float(block_amount)
        f_height = float(height)
        f_width = float(width)
        f_depth = float(depth)

        value_to_log = f_block_amount + f_block_amount * f_height + f_width * f_height * f_depth
        
        if value_to_log <= 0:
            print(f"  Warning ({filename_for_log}): The value for log10 calculation is non-positive ({value_to_log}). Unable to calculate difficulty_factor.")
            return None

        difficulty = math.log10(value_to_log) - 0.4
        return round(difficulty, 1)

    except KeyError as e:
        print(f"  Warning ({filename_for_log}): Missing key {e} during calculation. Unable to calculate difficulty_factor.")
        return None
    except TypeError as e:
        print(f"  Warning ({filename_for_log}): Type error during calculation: {e}. Please check if all values are numeric. Unable to calculate difficulty_factor.")
        return None
    except Exception as e:
        print(f"  Unexpected error occurred while calculating difficulty for {filename_for_log}: {e}")
        return None

def process_json_files(source_directory, destination_directory):
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory, exist_ok=True)
        print(f"Created target folder: {destination_directory}")

    for filename in os.listdir(source_directory):
        if filename.endswith(".json"):
            source_filepath = os.path.join(source_directory, filename)
            destination_filepath = os.path.join(destination_directory, filename)

            print(f"Processing file: {source_filepath}")

            try:
                with open(source_filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                print(f"  Error: Unable to decode JSON content in {filename}. Skipping.")
                continue
            except Exception as e:
                print(f"  Error: Failed to read {filename}: {e}. Skipping.")
                continue

            difficulty_factor_value = calculate_difficulty_factor(data, filename)
            
            output_data_dict = {}

            if difficulty_factor_value is not None:
                id_field_found = False
                for key, value in data.items():
                    output_data_dict[key] = value
                    if key == "id":
                        output_data_dict["difficulty_factor"] = difficulty_factor_value
                        id_field_found = True
                
                if id_field_found:
                    print(f"  Successfully calculated difficulty_factor: {difficulty_factor_value}. Added after 'id' field.")
                else:
                    print(f"  Warning: 'id' field not found in {filename}. 'difficulty_factor' ({difficulty_factor_value}) not added as required.")
            else:
                output_data_dict = data.copy()
                print(f"  The difficulty_factor for file {filename} could not be calculated. New field not added.")

            try:
                with open(destination_filepath, 'w', encoding='utf-8') as f:
                    json.dump(output_data_dict, f, indent=4, ensure_ascii=False)
                print(f"  Successfully processed and saved to: {destination_filepath}")
            except Exception as e:
                print(f"  Error: Failed to write to {destination_filepath}: {e}")

if __name__ == "__main__":
    # Please fill in a folder containing several newly generated JSON file architecture data.
    source_dir = ""
    destination_dir = ""
    
    process_json_files(source_dir, destination_dir)
    print("All JSON files have been processed.")
