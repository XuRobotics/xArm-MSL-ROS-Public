#!/bin/bash

log_file="repair_final.log"
> "$log_file"

echo "Scanning all 3.mp4 files and creating missing splits..."

# Find all 3.mp4 files safely
find videos -type f -name "3.mp4" -print0 | while IFS= read -r -d '' filepath; do
    if [[ ! -f "$filepath" ]]; then
        echo "Skipping invalid path: $filepath" >> "$log_file"
        continue
    fi

    dir=$(dirname "$filepath")
    base="$dir/3"
    left="${base}-left.mp4"
    right="${base}-right.mp4"

    # Skip if both LEFT and RIGHT already exist and are non-empty
    if [[ -s "$left" && -s "$right" ]]; then
        echo "Skipping: $filepath (already split)"
        continue
    fi

    # Create RIGHT if missing or empty
    if [[ ! -s "$right" ]]; then
        echo "Recreating RIGHT: $right"
        timeout 30 ffmpeg -loglevel error -y -i "$filepath" -filter:v "crop=iw*2/3:ih:iw/3:0" -c:a copy "$right" || {
            echo "RIGHT failed or timed out: $filepath" >> "$log_file"
        }
    fi

    # Create LEFT if missing or empty
    if [[ ! -s "$left" ]]; then
        echo "Recreating LEFT: $left"
        timeout 30 ffmpeg -loglevel error -y -i "$filepath" -filter:v "crop=iw*2/3:ih:0:0" -c:a copy "$left" || {
            echo "LEFT failed or timed out: $filepath" >> "$log_file"
        }
    fi
done

echo "Split pass complete. See '$log_file' for any issues."


# Count and summary section
#!/bin/bash

echo "Scanning for 3.mp4, 3-left.mp4, and 3-right.mp4 files..."

echo ""
echo "Found 3.mp4 files:"
echo "--------------------"
find videos -type f -name "3.mp4" | sort

echo ""
echo "Found 3-left.mp4 files:"
echo "--------------------------"
find videos -type f -name "3-left.mp4" | sort

echo ""
echo "Found 3-right.mp4 files:"
echo "---------------------------"
find videos -type f -name "3-right.mp4" | sort

echo ""
echo "Summary:"
echo "---------------------------"
echo "3.mp4      : $(find videos -type f -name "3.mp4" | wc -l)"
echo "3-left.mp4 : $(find videos -type f -name "3-left.mp4" | wc -l)"
echo "3-right.mp4: $(find videos -type f -name "3-right.mp4" | wc -l)"
echo "---------------------------"