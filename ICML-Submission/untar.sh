cd $1
cat * > "$1.tar.gz"
tar -xzvf "$1.tar.gz"
rm "$1.tar.gz"