## use this script to process the datasets for FAKE, RELIABLE AND CONSPIRACY data. 
# run the rogram with EXACTLY 5 argruments as shown below:
#       python <script> <input-file-name> <output-CSV-file> <#skipInitialRows> <#rowsToProcess> <#rowsAfterWhichToShowPrintMessageTracker>
#
#       #skipInitialRows
#               is an integer. It is the index value in the url columns from which the processing should start.
#       #rowsToProcess
#               is an integer. If = -1 then will read in all the rows after skipping as per above parameter.
#               Otherwise, will read in the number specified.
## note only if running in jupyter then uncomment the two lines for import nest_asyncio and the nest_asyncio.apply()

import pandas as pd
import twint
import csv
#import os
import sys
from datetime import datetime
import logging
#import nest_asyncio
#nest_asyncio.apply()

global dataIpFile, csvOpFile, skipInitialRows, rowsToProcess, printMessageFreq

def ProcessCommandLineArgs():
    global dataIpFile, csvOpFile, skipInitialRows, rowsToProcess, printMessageFreq
    ## check number of arguments passed, should be 5
    if len(sys.argv) != 6:
        print(f"ERROR: Expected EXACTLY 6 argruments as shown below:\n")
        print(f"python <script> <input-file-name> <output-CSV-file> <#skipInitialRows> <#rowsToProcess> <#rowsAfterWhichToShowPrintMessageTracker>")
        print(f"But you passed {len(sys.argv)} arguments.")
        print(f"ERROR: Exiting program with RC = 100")
        exit(100)
    ## pick up and assign the arguments
    try:
        dataIpFile = sys.argv[1]
        csvOpFile = sys.argv[2]
        skipInitialRows = int(sys.argv[3])
        rowsToProcess = int(sys.argv[4])
        printMessageFreq = int(sys.argv[5])
    except:
        print(f"ERROR: Some parameters are wrong...recheck and run again.\nExiting the program with RC = 110")
        exit(110)
    
    return(0)

######## main logic starts ##########
if ProcessCommandLineArgs() == 0:
    logFileName = sys.argv[0] + '_LOG_' + sys.argv[3] + '_' + sys.argv[4] + '_' + sys.argv[5] + '.log'
    logging.basicConfig(level=logging.WARNING, filename=logFileName,                  \
        filemode='w', format='%(asctime)s %(levelname)s:%(message)s')
    print(f"All command line arguments are valid....starting main processing")
    logging.warning(f"All command line arguments are valid....starting main processing.")

print(f'\nStart time: {datetime.now().strftime("%c")}')
logging.warning(f'\nStart time: {datetime.now().strftime("%c")}')

print(f"\n")
print(f"Processing with command line arguments as:")
print(f"1) dataIpFile       = {dataIpFile}")
print(f"2) csvOpFile        = {csvOpFile}")
print(f"3) skipInitialRows  = {skipInitialRows}")
print(f"4) rowsToProcess    = {rowsToProcess}")
print(f"5) printMessageFreq = {printMessageFreq}")
print(f"\n")
logging.warning(f"\n\n")
logging.warning(f"Processing with command line arguments as:")
logging.warning(f"1) dataIpFile       = {dataIpFile}")
logging.warning(f"2) csvOpFile        = {csvOpFile}")
logging.warning(f"3) skipInitialRows  = {skipInitialRows}")
logging.warning(f"4) rowsToProcess    = {rowsToProcess}")
logging.warning(f"5) printMessageFreq = {printMessageFreq}")
logging.warning(f"\n\n")

## read in the all the rows if rowsToProcess is -1 AFTER skipping the number of data rows specified.
if rowsToProcess == -1 :
    dfIpData = pd.read_csv(dataIpFile, skiprows = range(1, skipInitialRows + 1), sep = ',', usecols = ['url'], low_memory=False)
else:
    dfIpData = pd.read_csv(dataIpFile, skiprows = range(1, skipInitialRows + 1), nrows = rowsToProcess, sep = ',', usecols = ['url'], low_memory=False)
##dfIpData = pd.read_csv(dataFile, sep = ',', usecols = ['url'], low_memory=False)

print(f"Input data read into dataframe.\n")
logging.warning(f"Input data read into dataframe.\n")

colNames = ['id', 'conversation_id', 'created_at', 'date', 'timezone', 'place', 'tweet', 'hashtags', 'cashtags', 'user_id', 'user_id_str', 'username', 'name', 'day', 'hour', 'link', 'retweet', 'nlikes', 'nreplies', 'nretweets', 'quote_url', 'search', 'near', 'geo', 'source', 'user_rt_id', 'user_rt', 'retweet_id', 'reply_to', 'retweet_date']

## create a new file for the final output csv and simply write the header
with open(csvOpFile, 'w') as fOpCsv:
    writer = csv.DictWriter(fOpCsv, delimiter=',', lineterminator='\n',fieldnames=colNames)
    writer.writeheader()

c = twint.Config()
c.Hide_output = True
c.Pandas = True
counter = 0
for url in dfIpData['url']:
    counter += 1
    if counter % printMessageFreq == 0:
        print(f"\nProcessing row number {counter}")
    c.Search = url
    twint.run.Search(c)
    logging.warning(f"Completed Tweet Extraction for Row # {counter} ::: url = {url}")
    twint.storage.panda.Tweets_df.to_csv(csvOpFile, mode='a', index=False, header=False)
    logging.warning(f"Completed output file append for Row # {counter} ::: url = {url}")

print(f"Number of rows skipped   = {skipInitialRows}")
print(f"Number of rows processed = {counter}")
logging.warning(f"Number of rows skipped   = {skipInitialRows}")
logging.warning(f"Number of rows processed = {counter}")

print(f'\nEnd time: {datetime.now().strftime("%c")}')
logging.warning(f'\nEnd time: {datetime.now().strftime("%c")}')