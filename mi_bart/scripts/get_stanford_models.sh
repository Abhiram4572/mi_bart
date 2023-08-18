

#!/usr/bin/env sh
# This script downloads the Stanford CoreNLP models.

CORENLP=stanford-corenlp-full-2015-12-09
SPICELIB=/data1/abhiram/anaconda3/envs/vlt5/lib/python3.7/site-packages/language_evaluation/coco_caption_py3/pycocoevalcap/spice/lib
JAR=stanford-corenlp-3.6.0

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

if [ -f $SPICELIB/$JAR.jar ]; then
  echo "Found Stanford CoreNLP."
else
  echo "Downloading..."
  wget http://nlp.stanford.edu/software/$CORENLP.zip
  echo "Unzipping..."
  unzip $CORENLP.zip -d $SPICELIB/
  mv $SPICELIB/$CORENLP/$JAR.jar $SPICELIB/
  mv $SPICELIB/$CORENLP/$JAR-models.jar $SPICELIB/
  rm -f $CORENLP.zip
  rm -rf $SPICELIB/$CORENLP/
  echo "Done."
fi