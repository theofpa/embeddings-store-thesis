#!/usr/bin/bash
export TF_CPP_MIN_LOG_LEVEL=3
export TF_FORCE_GPU_ALLOW_GROWTH=true
for detector in KS LSDD MMD Classifier
do
  for h_size in `seq 1000 1000 10000`
  do
    for seed in `seq 1 5`
    do
      for test_set in h0 h1
      do
        python drift-detection.py with \
        detector=$detector \
        ref_size=10000 \
        h_size=$h_size \
        seed=$seed \
        dataset=amazon_us_reviews \
        subset=Books_v1_02 \
        drift_attribute=star_rating \
        drift_attribute_value=4 \
        test_set=$test_set \
        text_field=review_body
      done
    done
  done
done
#
#for detector in MMD KS LSDD Classifier
#do
#  for h_size in `seq 1000 1000 10000`
#  do
#    for seed in `seq 1 5`
#    do
#      for test_set in h0 h1
#      do
#        python drift-detection.py with \
#        detector=$detector \
#        ref_size=10000 \
#        h_size=$h_size \
#        seed=$seed \
#        test_set=$test_set \
#      done
#    done
#  done
#done
