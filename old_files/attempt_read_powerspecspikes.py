with open('.\\Probe07_METIS_powerspectra.csv') as f:
    clicks1, clicks2, clicks3, clicks4 = [], [], [], []
    for i, line in enumerate(f):
        nums = line.split(',')
        nums1 = nums[1]
        nums2 = nums[2]
        print(nums1, nums2)
        if line[0] == 'X':
            clicks1.append(float(nums1))
            clicks1.append(float(nums2))
        if line[0] == 'Y':
            clicks2.append(float(nums1))
            clicks2.append(float(nums2))
        if line[0] == 'Z':
            clicks3.append(float(nums1))
            clicks3.append(float(nums2))
        if line[0] == 'T':
            clicks4.append(float(nums1))
            clicks4.append(float(nums2))

print(clicks1)
print(clicks2)
print(clicks3)
print(clicks4)