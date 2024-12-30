import os
from secedgar import filings, FilingType
script_dir = os.path.dirname(os.path.abspath(__file__))
pdfs_dir = os.path.join(script_dir, "pdfs")

my_filings = filings(cik_lookup=["aapl"],
                     filing_type=FilingType.FILING_10Q,
                     count=1,
                     user_agent="example example@ucsc.edu")
print(my_filings)
my_filings.save("./pdfs")
print("pdfs extracted and saved")