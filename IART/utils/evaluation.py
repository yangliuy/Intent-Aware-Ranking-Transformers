import sys;
from douban_evaluation import mean_average_precision

def get_p_at_n_in_m(data, n, m, ind):
	pos_score = data[ind][0];
	curr = data[ind:ind+m];
	curr = sorted(curr, key = lambda x:x[0], reverse=True)

	if curr[n-1][0] <= pos_score:
		return 1;
	return 0;

def get_map(data, m, ind):
	curr = data[ind:ind + m];
	sort_data = sorted(curr, key=lambda x: x[0], reverse=True)
	m_a_p = mean_average_precision(sort_data)
	return m_a_p

def evaluate(file_path):
	data = []
	with open(file_path, 'r') as file:
		for line in file:
			line = line.strip();
			tokens = line.split("\t")
		
			if len(tokens) != 2:
				continue
		
			data.append((float(tokens[0]), int(tokens[1])));
		
	#assert len(data) % 10 == 0
	
	p_at_1_in_2 = 0.0
	p_at_1_in_10 = 0.0
	p_at_2_in_10 = 0.0
	p_at_5_in_10 = 0.0
	map_sum = 0.0

	length = len(data)/10 # number of queries
	print('num of queries: ', length)

	for i in xrange(0, length):
		ind = i * 10 # use ind to index the first doc of each query
		assert data[ind][1] == 1
	
		p_at_1_in_2 += get_p_at_n_in_m(data, 1, 2, ind)
		p_at_1_in_10 += get_p_at_n_in_m(data, 1, 10, ind)
		p_at_2_in_10 += get_p_at_n_in_m(data, 2, 10, ind)
		p_at_5_in_10 += get_p_at_n_in_m(data, 5, 10, ind)
		map_sum += get_map(data, 10, ind)
		# add MAP here for IADAM evaluation



	return (p_at_1_in_2/length, p_at_1_in_10/length, p_at_2_in_10/length,
			p_at_5_in_10/length, map_sum/length)


if __name__ == '__main__':
	if len(sys.argv) < 2:
		print("plean input parameters: score_file")
		sys.exit(1)
	result = evaluate(sys.argv[1])
	for r in result:
		print(r)
	# m_line = "\t".join([str(m) for m in result])
	# print('[current metrics (r2@1 r10@1 r10@2 r10@5 map)]\t', m_line)
	print('[current metrics (r2@1 r10@1 r10@2 r10@5 map)]\t{:f}\t{:f}\t{:f}\t{:f}\t{:f}'.format(
		result[0], result[1], result[2], result[3], result[4]))