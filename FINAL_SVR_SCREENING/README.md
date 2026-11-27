# Baseline SVR natural-product screening
Existing predictions from screen_np_extended.py; no new fitting or prediction.
Original historical SVR models for CDK2, CDK4 and CDK6, not optimised SVR models.
Known compounds in any curated target dataset excluded. Joint AD = minimum of three target-specific nearest-training-neighbour similarities.
Dual hit: both predicted pIC50 >= 6. Predicted selectivity fold = 10**(min(CDK2,CDK4 pIC50)-CDK6 pIC50).
Candidates are computational hypotheses, not experimentally verified.
If gzip files are split, concatenate the numbered parts in order to reconstruct the original gzip stream.
