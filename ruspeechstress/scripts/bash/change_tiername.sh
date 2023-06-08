# changes tier names in TextGrid to fit for Prosogram
sed -i 's/"MAS"/"syll"/g' "$1"
sed -i 's/"MAU"/"segm"/g' "$1"