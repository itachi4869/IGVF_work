import numpy as np
import pandas as pd

bed_file_path = 'all_cCREs.bed'
bed_df = pd.read_csv(bed_file_path, sep=' ', header=None)
bed_df.columns = ['chr', 'start', 'end']
n_cCREs = len(bed_df)

cnt_no_range = 0

def select_random_ranges(row1, row2, range_length=300):

    global cnt_no_range
    chrom, start, end = row1['chr'], int(row1['end']) + 1, int(row2['start']) - 1
    if end - start <= range_length: # cannot select a range if the gap is less than the range length
        cnt_no_range += 1
        return []
    elif end - start <= range_length * 3 + 1: # if gap is less than 3 times the range length, only one range will be selected, otherwise two ranges will be selected
        num_ranges = 1
    else:
        num_ranges = 2 # select two ranges
    
    ranges = []

    if num_ranges == 1:
        random_start = np.random.randint(start, end - range_length)
        random_end = random_start + range_length
        anno = row1['chr'] + ':' + str(random_start) + '-' + str(random_end)
        ranges.append((chrom, random_start, random_end, anno))
    elif num_ranges == 2:
        while True:
            random_start1 = np.random.randint(start, end - range_length * 2 - 2)
            random_end1 = random_start1 + range_length
            random_start2 = np.random.randint(random_end1 + 1, end - range_length)
            random_end2 = random_start2 + range_length
            if random_end2 <= end:
                anno1 = row1['chr'] + ':' + str(random_start1) + '-' + str(random_end1)
                anno2 = row1['chr'] + ':' + str(random_start2) + '-' + str(random_end2)
                ranges.append((chrom, random_start1, random_end1, anno1))
                ranges.append((chrom, random_start2, random_end2, anno2))
                break
    
    return ranges

# Apply the function to each row and collect the results
random_ranges = []
for i in range(n_cCREs - 1):
    row1 = bed_df.iloc[i]
    row2 = bed_df.iloc[i + 1]
    if row1['chr'] != row2['chr']:
        continue
    random_ranges.extend(select_random_ranges(row1, row2))

# Convert the results to a DataFrame
random_ranges_df = pd.DataFrame(random_ranges, columns=['chrom', 'start', 'end', 'anno'])

indices = np.arange(len(random_ranges_df))
np.random.shuffle(indices)
selected_indices = indices[:n_cCREs]
selected_indices.sort()
print(cnt_no_range)
new_random_ranges_df = random_ranges_df.iloc[selected_indices]

output_file_path = 'all_non_cCREs.bed'
new_random_ranges_df.to_csv(output_file_path, sep='\t', header=False, index=False)