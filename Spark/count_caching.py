input_repository = '/home/alog0/taeha/Spark/Count_caching/input/All_uncache_1'

count = 0
with open(input_repository, 'r', encoding='utf-8-sig') as data_file:
	while True:
		line = data_file.readline()
		if not line:
			break
		line_split = line.split(':')
		path = line_split[0]
		num = line_split[1]
		count += int(num)
		if int(num) != 0:
			print(line)
print("\nAll : ",count)
