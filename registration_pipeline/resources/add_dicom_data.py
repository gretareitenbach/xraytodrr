import pydicom
import argparse

'''
Usage:

python add_dicom_data.py \
    --i /path/to/original/xray.dcm \
    --o /path/to/updated/xray_with_sdd.dcm \
    --sdd 1020.0 

'''

def main():
    """
    Adds or updates the Source-to-Detector Distance (SDD) tag in a DICOM file.
    """
    parser = argparse.ArgumentParser(
        description="Adds or updates the Source-to-Detector Distance (0x0018, 0x1110) in a DICOM file."
    )
    
    parser.add_argument(
        "-i", "--input_file",
        type=str,
        required=True,
        help="Path to the input DICOM file (.dcm)."
    )

    parser.add_argument(
        "-o", "--output_file",
        type=str,
        required=True,
        help="Path to save the new, updated DICOM file."
    )
    
    parser.add_argument(
        "--sdd",
        type=float,
        required=True,
        help="Source-to-Detector Distance (SDD) value to write to the DICOM header."
    )
    
    args = parser.parse_args()

    ds = pydicom.dcmread(args.input_file)
    ds[(0x0018, 0x1110)] = pydicom.DataElement((0x00181110), 'DS', str(args.sdd))
    ds.save_as(args.output_file)

    print(f"Successfully added SDD={args.sdd} to '{args.input_file}'")
    print(f"New file saved to: '{args.output_file}'")

if __name__ == "__main__":
    main()