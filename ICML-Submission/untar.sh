cd $1
cat * > "model.tar.gz"
tar -xzvf "model.tar.gz"
rm "model.tar.gz"