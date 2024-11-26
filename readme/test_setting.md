# Key Settings That Affect the Results
Note that there are two settings for the DSEC dataset, and the results presented in the paper are based on these settings.
We are not claiming that these two settings must be fixed, and they may seem somewhat arbitrary. However, if you intend to
use our method for comparison, you are free to adjust these settings. Just be sure to compare DEOE under the same settings
to maintain fairness.

## Data Frequency of DSEC-Detection 
The data frequency is set to 40 Hz in our experiments, considering the characteristics of the event stream, i.e., its high
temporal resolution. The annotation frequency is 20 Hz, as described in in [DSEC-Detection](https://dsec.ifi.uzh.ch/dsec-detection/). Results under 20 Hz data frequency 
is better than results under 40 Hz data frequency, since the memory networks suffer less burden. You can choose do experiments with a
20 Hz data, that's Ok but you should test DEOE on the same dataset.

## Small Box Filter
This strategy of filtering small boxes is commonly used in GEN1 and GEN4 datasets, so we apply it to the DSEC-Detection dataset as well,
since its annotations are pseudo-labels. Our settings are as follows:
```python
    if apply_bbox_filters:
        min_box_diag = 40
        min_box_side = 30
        if downsampled_by_2:
            assert min_box_diag % 2 == 0
            min_box_diag //= 2
            assert min_box_side % 2 == 0
            min_box_side //= 2
```
This setting has a significant impact on the final results. You may also experiment with other values for `min_box_diag` and `min_box_side`. However,
to ensure fairness, be sure to test DEOE using the same settings.



