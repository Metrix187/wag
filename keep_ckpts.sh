#!/usr/bin/env bash
# grab the epoch-boundary checkpoints before save_total_limit=3 rotates them off.
#
# 8077 rows / effective batch 16 = ~505 steps an epoch, so 500 and 1000 are the ends of
# epochs 1 and 2. the trainer only ever keeps the last three checkpoints, which by the end
# of the run means ~1300/1400/1500 — all of them deep in epoch 3. that's exactly the wrong
# set to be holding if epoch 3 turns out to overfit, and the only way back would be
# retraining. so: copy the two that matter out of the rotation's reach.
#
# runs from the colab terminal so it doesn't touch the kernel the training is sitting in.
set -u

WAG=/content/drive/MyDrive/wag
CK=$WAG/ckpt-v2

# a checkpoint dir shows up the moment saving starts, so copying on sight gets you half a
# file. wait for its size to stop moving instead.
settled() {
  local d=$1 a b
  a=$(du -s "$d" 2>/dev/null | cut -f1)
  sleep 25
  b=$(du -s "$d" 2>/dev/null | cut -f1)
  [ -n "$a" ] && [ "$a" = "$b" ] && [ "$a" -gt 100000 ]
}

for step in 500 1000; do
  dest=$WAG/keep-step$step
  if [ -d "$dest" ]; then
    echo "$(date +%H:%M:%S) keep-step$step is already here, skipping"
    continue
  fi
  echo "$(date +%H:%M:%S) waiting for checkpoint-$step"
  while [ ! -d "$CK/checkpoint-$step" ]; do sleep 45; done
  echo "$(date +%H:%M:%S) checkpoint-$step appeared, waiting for the write to finish"
  until settled "$CK/checkpoint-$step"; do sleep 15; done
  cp -r "$CK/checkpoint-$step" "$dest"
  echo "$(date +%H:%M:%S) kept -> $dest  ($(du -sh "$dest" | cut -f1))"
done

echo "$(date +%H:%M:%S) both saved, nothing left to watch"
