import pandas as pd
import re
import os

def extract_number(value):
    """
    Extracts the numeric value from a string like '13.123,56 KGM' or '50,5 KG'.
    Converts 'TON' to 1000 KG and treats 'KILOGRAMS' the same as 'KG' or 'KGM'.
    Returns None if the value is '-' or doesn't match the pattern.
    """
    if value == "-":
        return None
    # Match numbers with commas and dots (e.g., 13.123,56 or 50,5)
    match = re.match(r"([\d.,]+)\s*(KGM|KG|KILOGRAMS|TON)", str(value), re.IGNORECASE)
    if match:
        number_str = match.group(1).replace(",", "")
        unit = match.group(2).upper()

        # Convert TON to KG (1 TON = 1000 KG)
        if unit == "TON":
            return float(number_str) * 1000
        elif unit in ["CT", "PAIL"]:
            return float(number_str) * 2 * 24 * 280 / 1000
        elif unit == "PK":
            return float(number_str) * 12 * 425 / 1000
        elif unit == "BARREL":
            return float(number_str) * 30 * 180 / 1000
        elif unit == "BOX/BAG/PACK":
            return float(number_str) * 24 * 235 / 1000
        # Treat KILOGRAMS the same as KG or KGM
        elif unit in ["KILOGRAMS", "KG", "KGM"]:
            return float(number_str)
    return None


def clean_data(df):
    # Remove rows where 'Value (USD)' is '-'
    df = df[df["Value (USD)"] != "-"]

    # Process 'Weight' and 'Quantity' columns
    for index, row in df.iterrows():
        weight_value = extract_number(row["Weight"])
        quantity_value = extract_number(row["Quantity"])

        # If 'Weight' is missing, use 'Quantity' if available
        if pd.isna(weight_value) or weight_value is None:
            if quantity_value is not None:
                df.at[index, "Weight"] = quantity_value
            else:
                # Drop the row if both 'Weight' and 'Quantity' are missing
                df.drop(index, inplace=True)
        else:
            df.at[index, "Weight"] = weight_value

    # Calculate 'Unit Price' as Value (USD) / Weight
    df["Unit Price"] = df["Value (USD)"] / df["Weight"]

    return df


def process(input_file):
    """
    Processes the input Excel file, cleans the data, and exports the results to a new file.
    """
    # Read the Excel file with all sheets
    all_sheets = pd.read_excel(input_file, sheet_name=None)  # Read all sheets into a dictionary

    # Clean each sheet and store the cleaned data in a new dictionary
    cleaned_sheets = {}
    for sheet_name, df in all_sheets.items():
        cleaned_df = clean_data(df)

        # Add a row for the average formula
        average_row = pd.DataFrame({"No.": ["Average"], "Unit Price": [None]})
        cleaned_df = pd.concat([cleaned_df, average_row], ignore_index=True)

        cleaned_sheets[sheet_name] = cleaned_df

    # Create the output file path
    output_file = os.path.splitext(input_file)[0] + "_processed.xlsx"

    # Export all cleaned sheets to a new Excel file
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        for sheet_name, df in cleaned_sheets.items():
            # Write the DataFrame to Excel
            df.to_excel(writer, sheet_name=sheet_name, index=False)

            # Access the Excel sheet and add the formula for the average
            workbook = writer.book
            worksheet = writer.sheets[sheet_name]

            # Determine the last row and column for the 'Unit Price' column
            last_row = len(df)
            unit_price_col = df.columns.get_loc("Unit Price") + 1  # Excel is 1-indexed

            # Add the AVERAGE formula to the last row of the 'Unit Price' column
            formula = f"=AVERAGE({worksheet.cell(row=2, column=unit_price_col).coordinate}:{worksheet.cell(row=last_row, column=unit_price_col).coordinate})"
            worksheet.cell(row=last_row + 1, column=unit_price_col, value=formula)

    print(f"Data cleaned and exported to {output_file}")


# Example usage
process("export.xlsx")
