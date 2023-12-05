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


def merge_slim_files(folder_path, output_path):

    # Get a list of all CSV files in the folder
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

    # Check if there are any CSV files in the folder
    if not csv_files:
        print(f"Error: No CSV files found in '{folder_path}'.")
        return

    # Read the first CSV file to get the column names
    first_file_path = os.path.join(folder_path, csv_files[0])
    first_df = pd.read_csv(first_file_path)

    # Create an empty DataFrame with the same columns
    merged_df = pd.DataFrame(columns=first_df.columns)

    # Iterate through all CSV files and append them to the merged DataFrame
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        df = pd.read_csv(file_path)
        merged_df = pd.concat([merged_df, df], ignore_index=True)

    # Save the merged DataFrame to a new CSV file
    merged_df.to_csv(output_path, index=False)

    print(f"Merged CSV files saved to '{output_path}'.")


if __name__ == "__main__":
    # Set the paths to your raw_data and raw_data_slim folders
    raw_data_folder = "raw_data"
    raw_data_slim_folder = "raw_data_slim"
    #folder_path = '/path/to/your/csv/files'
    output_path = 'raw_data_slim/merged_thai_restaurant_file.csv'
    # Process files in raw_data and save them to raw_data_slim
    process_files(raw_data_folder, raw_data_slim_folder)
    merge_slim_files(raw_data_slim_folder, output_path)
