# Data distribution analysis

Frozen protocol `48054c137c8e093957c57eef6d761750cfac6f7b437e089cebe42053fb4b62ab`. Train: 138 images, 81 Flash and 93 Black instances. At imgsz 1536, 173/174 targets have at least one side <16 px and 52/174 <8 px (also <1 P3 cell). Among 93 Black instances, 33 lie in the joint train-median small/low-contrast quadrant. See `gt_statistics.csv` and `difficulty_distributions.png`. Validation statistics are descriptive only.
