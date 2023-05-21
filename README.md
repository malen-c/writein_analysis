ANALYSIS OF WRITE IN BALLOTS FOR 2020 MULTNOMAH COUNTY MAYORAL ELECTION
Malen Cuturic:)

A. Reading of Ballots:

    - Mount optical drive which contains scan of ballots. Path to folder containing scans will be referred to as $OPTICAL$.

    - Main file to be used is "reader.py", which must be edited so that the image_folder variable is $OPTICAL$ (glad we came up with that shorthand!).
 
    - File structure should look like:

          >Analysis
              >templates        (contains "template" images against which ballots of the same style are aligned)
                  >...
              >funs.py          (utilities used by reader.py)
              >reader.py        (main file)
              >writeins.csv     (file containing Ballot ID and Ballot Style for all write ins)
              >results.csv      (empty csv into which results are encoded by reader.py)

      Other files are unnecessary for reading of ballots.

    - After doing these configurations, you can run reader.py. It will split the ballots into 4700 equal sized groups, and update you each time
      it reads all of these groups. This could take kind of a while! It ran for about one day on my computer, but that was mostly limited by
      the amount of time it took to download the ballots from the drive, so I'd anticipate it being faster than that with the optical drive.