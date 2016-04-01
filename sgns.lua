-- A torch-based implementation of skip-gram with negative samping
-- Not exactly the same but follows many of the original implementation at http://word2vec.googlecode.com/svn/trunk/
-- Initially implemented by Joo-Kyung Kim (kimjook@cse.ohio-state.edu, supakjk@gmail.com)


require('sys')
require('paths')
require('math')
require('nn')

torch.setdefaulttensortype('torch.FloatTensor')

cmd = torch.CmdLine()
cmd:option('-read_vocab', '', 'load word list file (decreasing frequency order)')
cmd:option('-save_vocab', '', 'save word list file (decreasing frequency order). only works when -read_vocab is not set')
cmd:option('-train', '', 'training corpus file')
cmd:option('-window', 5, 'context window size')
cmd:option('-size', 300, '# hidden units')
cmd:option('-seed', 1, 'seed')
cmd:option('-negative', 3, '# negative samples')
cmd:option('-alpha', 0.025, 'initial learning rate')
cmd:option('-iter', 5, '# epochs')
cmd:option('-sample', 1e-5, 'set threshold for occurrence of words (frequent words will be randomly down-sampled)')
cmd:option('-min_count', 0, 'minimum word frequencies (words occuring less than this is skipped)')
cmd:option('-out_dir', '', 'the directory weights will be saved')

local params = cmd:parse(arg)

-- setting the weight output directory
if params.out_dir == '' then
	io.stderr:write('-out_dir is missed\n')
	os.exit()
end
if paths.dir(params.out_dir) ~= nil then
	io.stderr:write(params.out_dir .. ' already exists\n')
	os.exit()
end

paths.mkdir(params.out_dir)
--~

if params.train == '' then
	io.stderr:write('-train is missed')
	os.exit()
end


local vocab_idxs = {}
local word_freq = torch.LongTensor()
local file
local vocab_cnt = 0
local corpus_word_cnt


--- get the vocabulary index and word frequencies from the corpus
function get_vocab_wordfreq()
	print('Reading the word frequencies: ' .. params.train)
	print(os.date())

	if params.read_vocab ~= '' then	-- read vocab indices
		file = io.open(params.read_vocab, 'r')
		while true do
			local line = file:read()
			if line == nil then break end
			vocab_cnt = vocab_cnt + 1
			vocab_idxs[line] = vocab_cnt
		end
		file:close()
		--~

		-- get the word frequencies from the corpus
		word_freq:resize(vocab_cnt):zero()
		file = io.open(params.train, 'r')
		local line_cnt = 0
		while true do
			local line = file:read()
			if line == nil then break end
			for v in string.gmatch(line, "%S+") do
				local cur_vocab_idx = vocab_idxs[v]
				word_freq[cur_vocab_idx] = word_freq[cur_vocab_idx] + 1
			end
			line_cnt = line_cnt + 1
		end
		file:close()
		word_freq[1] = line_cnt -- # </s> should be # sentences

		corpus_word_cnt = 0
		for i=1,word_freq:size(1) do
			corpus_word_cnt = corpus_word_cnt + word_freq[i]
		end
	elseif params.save_vocab ~= '' then
		file = io.open(params.train, 'r')
		local line_cnt = 0
		while true do
			local line = file:read()
			if line == nil then break end
			for v in string.gmatch(line, "%S+") do
				if vocab_idxs[v] == nil then
					vocab_idxs[v] = 1
				else
					vocab_idxs[v] = vocab_idxs[v] + 1
				end
			end
			line_cnt = line_cnt + 1
		end
		file:close()

		local temp_array = {}
		for cur_word, freq in pairs(vocab_idxs) do
			if freq >= params.min_count then
				temp_array[#temp_array+1] = {freq, cur_word}
				vocab_cnt = vocab_cnt + 1
			end
		end
		vocab_cnt = vocab_cnt + 1	-- there should be one more as a place holder for "</s>"

		function sort_comparator(x, y)
			return x[1] > y[1]
		end

		table.sort(temp_array, sort_comparator)

		word_freq:resize(vocab_cnt)
		word_freq[1] = line_cnt
		vocab_idxs['</s>'] = 1
		corpus_word_cnt = line_cnt

		file = io.open(params.save_vocab, 'w')
		file:write('</s>')
		for k,v in ipairs(temp_array) do
			word_freq[k+1] = v[1]
			corpus_word_cnt = corpus_word_cnt + v[1]
			vocab_idxs[v[2]] = k+1
			file:write('\n' .. v[2])
		end
		file:close()
		temp_array = nil
	else
		io.stderr:write('Either -read_vocab or -save_vocab must be set\n')
		os.exit()
	end
end

get_vocab_wordfreq()


-- set the list for word sampling
local SAMPLE_LIST_SIZE = 1e8
local sample_list = torch.IntTensor(SAMPLE_LIST_SIZE):zero()	-- for sampling from a unigram distribution
local train_words_pow = torch.LongTensor(1):zero()
local power = 0.75
for i=1,word_freq:size(1) do
	train_words_pow:add(torch.pow(word_freq[i], power))
end
local curIdx = 1
local d1 = torch.pow(word_freq[curIdx], power) / train_words_pow[1]
for a=1,sample_list:size(1) do
	sample_list[a] = curIdx
	if (a-1) / sample_list:size(1) > d1 then
		curIdx = curIdx + 1
		d1 = d1 + torch.pow(word_freq[curIdx], power) / train_words_pow[1]
	end
	if curIdx > word_freq:size(1) then
		curIdx = word_freq:size(1)
	end
end
--~

-- Exponent table
local MAX_EXP = 6
local EXP_TABLE_SIZE = 1000
local expTable = torch.FloatTensor(EXP_TABLE_SIZE+1)
for i=1,EXP_TABLE_SIZE do
	expTable[i] = math.exp(((i-1)/EXP_TABLE_SIZE*2 - 1) * MAX_EXP)
	expTable[i] = expTable[i] / (expTable[i]+1)
end
--~

-- variables
local word_mat = nn.LookupTable(vocab_cnt, params.size)
word_mat.weight:uniform(-0.5/params.size, 0.5/params.size)
local target_mat = nn.LookupTable(vocab_cnt, params.size)
target_mat.weight:zero()
target_mat.gradWeight = nil

local sampled_sent = torch.IntTensor(2000)

local word_cnt = 0
local last_word_cnt = 0
local word_cnt_actual = 0

local alpha = params.alpha
local best_loss = math.huge

local target
local label
local g	-- grad
local f
--~

torch.manualSeed(params.seed)

collectgarbage()


-- return a sentence tensor excluding sampled words from the sentence (proportional to # occurences)
function sample_sent_from_line(line, seq_tensor)
	seq_tensor:resize(2000)
	local sampled_sent_idx = 0
	if params.sample > 0 then
		for cur_word in string.gmatch(line, "%S+") do
			local cur_word_idx = vocab_idxs[cur_word]
			if cur_word_idx ~= nil then
				local ran = (math.sqrt(word_freq[cur_word_idx] / (params.sample * corpus_word_cnt)) + 1) * params.sample * corpus_word_cnt / word_freq[cur_word_idx]
				if ran >= torch.uniform() then
					sampled_sent_idx = sampled_sent_idx + 1
					seq_tensor[sampled_sent_idx] = cur_word_idx
				end
			end
		end
		if sampled_sent_idx <= 0 then sampled_sent_idx = 1 end	-- not to allow sampled_sent to be 0 dimensional
	else
		for cur_word in string.gmatch(line, "%S+") do
			local cur_idx = vocab_idxs[cur_word]
			if cur_idx ~= nil then
				sampled_sent_idx = sampled_sent_idx + 1
				seq_tensor[sampled_sent_idx] = cur_idx
			end
		end
	end
	seq_tensor:resize(sampled_sent_idx)
end


-- lower the learning rate
function adjust_alpha()
	word_cnt_actual = word_cnt_actual + word_cnt - last_word_cnt
	last_word_cnt = word_cnt
	alpha = params.alpha * (1 - word_cnt_actual / (params.iter * corpus_word_cnt + 1))
	if alpha < params.alpha * 0.0001 then alpha = params.alpha * 0.0001 end
end


-- optimization with negative sampling
function negative_sampling(input_word, v)
	local input_weight = word_mat.weight[input_word]
	local input_gradWeight = word_mat.gradWeight[input_word]
	local cur_epoch_loss = 0

	for j=1,params.negative+1 do	-- for each correct target and negative sampled target
		if j == 1 then
			target = sampled_sent[v]
			label = 1
		else
			target = sample_list[torch.random(SAMPLE_LIST_SIZE)]
			if target == 1 then target = torch.random(word_freq:size(1)-1) + 1 end
			if target == sampled_sent[v] then goto NS_LOOP_END end
			label = 0
		end

		local target_weight = target_mat.weight[target]

		f = input_weight * target_weight
		if f > MAX_EXP then
			g = (label-1) * alpha
		elseif f < -MAX_EXP then
			g = label * alpha
		else
			g = (label - expTable[math.floor( (f+MAX_EXP) * (EXP_TABLE_SIZE/MAX_EXP/2) ) + 1]) * alpha
		end
		input_gradWeight:add(g, target_weight)
		target_weight:add(g, input_weight)
		cur_epoch_loss = cur_epoch_loss - g / alpha

		::NS_LOOP_END::
	end

	input_weight:add(input_gradWeight)
	input_gradWeight:zero()
	return cur_epoch_loss
end


for epoch=1,params.iter do
	print('epoch: ' .. epoch)
	local epoch_loss = 0

	file = io.open(params.train, 'r')
	line_cnt = 0
	while true do	-- loop for each sentence
		local line = file:read()
		if line == nil then break end

		sample_sent_from_line(line, sampled_sent)

		if sampled_sent:size(1) >= 2 then
			if word_cnt - last_word_cnt > 10000 then	-- lower the learning rate periodically
				adjust_alpha()
			end

			for i=1,sampled_sent:size(1) do
				local input_word = sampled_sent[i]
				local cur_win_size = torch.random(params.window)
				local target_begin = math.max(i-cur_win_size, 1)
				local target_end = math.min(sampled_sent:size(1), i+cur_win_size)
				for v=target_begin,target_end do	-- for each context word
					if v ~= i then
						epoch_loss = epoch_loss + negative_sampling(input_word, v)
					end
				end

				word_cnt = word_cnt + 1
			end
		end

		line_cnt = line_cnt + 1
	end

	file:close()

	word_cnt_actual = word_cnt_actual + word_cnt - last_word_cnt
	word_cnt = 0
	last_word_cnt = 0
	print('gradient sum: ' .. epoch_loss)
	print(os.date())

	if epoch_loss < best_loss then
		best_loss = epoch_loss
	end

	torch.save(params.out_dir .. '/weight_' .. epoch .. '.th', word_mat.weight)
end

print('least gradient sum: ' .. best_loss)
