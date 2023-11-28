import os
import pandas as pd

def process_file(input_path, output_folder):
    # Define the columns you want to keep and their order
    desired_columns = [
        "placeId",
        "title",
        "reviewId",
        "reviewerId",
        "isLocalGuide",
        "likesCount",
        "publishedAtDate",
        "responseFromOwnerDate",
        "responseFromOwnerText",
        "reviewContext/Meal type",
        "reviewContext/Parking space",
        "reviewContext/Price per person",
        "reviewContext/Service",
        "reviewDetailedRating/Atmosphere",
        "reviewDetailedRating/Food",
        "reviewDetailedRating/Service",
        "reviewerNumberOfReviews",
        "text",
        "textTranslated",
        "stars"
    ]

  # Read the CSV file
    df = pd.read_csv(input_path)

    # Keep only the desired columns
    df = df[desired_columns]

    # Extract the filename (without extension) from the input path
    filename_without_extension = os.path.splitext(os.path.basename(input_path))[0]

    # Generate the output path with the "_slim" suffix
    output_path = os.path.join(output_folder, f"{filename_without_extension}_slim.csv")

    # Save the processed DataFrame to the output path
    df.to_csv(output_path, index=False)

def process_files(input_folder, output_folder):
    # Create the output folder if it doesn't exist

    # Iterate through each file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            input_path = os.path.join(input_folder, filename)

            # Process the file and save the result
            process_file(input_path, output_folder)
            print(f"Processed: {filename}")

if __name__ == "__main__":
    # Set the paths to your raw_data and raw_data_slim folders
    raw_data_folder = "raw_data"
    raw_data_slim_folder = "raw_data_slim"

    # Process files in raw_data and save them to raw_data_slim
    process_files(raw_data_folder, raw_data_slim_folder)
