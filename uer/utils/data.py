import os
import random
import pickle
import torch
from multiprocessing import Pool
from uer.utils.constants import *
from uer.utils.tokenizers import *
from uer.utils.misc import count_lines
from uer.utils.seed import set_seed


def mask_seq(src, tokenizer, whole_word_masking, span_masking, span_geo_prob, span_max_length):
    vocab = tokenizer.vocab

    for i in range(len(src) - 1, -1, -1):
        if src[i] != PAD_ID:
            break
    src_no_pad = src[:i + 1]

    tokens_index, src_no_pad = create_index(src_no_pad, tokenizer, whole_word_masking, span_masking, span_geo_prob, span_max_length)
    if len(src_no_pad) < len(src):
        src = src_no_pad + (len(src) - len(src_no_pad)) * [PAD_ID]
    else:
        src = src_no_pad

    random.shuffle(tokens_index)
    num_to_predict = max(1, int(round(len(src_no_pad) * 0.15)))
    tgt_mlm = []
    for index_set in tokens_index:
        if len(tgt_mlm) >= num_to_predict:
            break
        if whole_word_masking:
            i = index_set[0]
            mask_len = index_set[1]
            if len(tgt_mlm) + mask_len > num_to_predict:
                continue

            for j in range(mask_len):
                token = src[i + j]
                tgt_mlm.append((i + j, token))
                prob = random.random()
                if prob < 0.8:
                    src[i + j] = vocab.get(MASK_TOKEN)
                elif prob < 0.9:
                    while True:
                        rdi = random.randint(1, len(vocab) - 1)
                        if rdi not in [vocab.get(CLS_TOKEN), vocab.get(SEP_TOKEN), vocab.get(MASK_TOKEN), PAD_ID]:
                            break
                    src[i + j] = rdi
        elif span_masking:
            i = index_set[0]
            span_len = index_set[1]
            if len(tgt_mlm) + span_len > num_to_predict:
                continue

            for j in range(span_len):
                token = src[i + j]
                tgt_mlm.append((i + j, token))
            prob = random.random()
            if prob < 0.8:
                for j in range(span_len):
                    src[i + j] = vocab.get(MASK_TOKEN)
            elif prob < 0.9:
                for j in range(span_len):
                    while True:
                        rdi = random.randint(1, len(vocab) - 1)
                        if rdi not in [vocab.get(CLS_TOKEN), vocab.get(SEP_TOKEN), vocab.get(MASK_TOKEN), PAD_ID]:
                            break
                    src[i + j] = rdi
        else:
            i = index_set[0]
            token = src[i]
            tgt_mlm.append((i, token))
            prob = random.random()
            if prob < 0.8:
                src[i] = vocab.get(MASK_TOKEN)
            elif prob < 0.9:
                while True:
                    rdi = random.randint(1, len(vocab) - 1)
                    if rdi not in [vocab.get(CLS_TOKEN), vocab.get(SEP_TOKEN), vocab.get(MASK_TOKEN), PAD_ID]:
                        break
                src[i] = rdi
    tgt_mlm = sorted(tgt_mlm, key=lambda x: x[0])
    return src, tgt_mlm


def create_index(src, tokenizer, whole_word_masking, span_masking, span_geo_prob, span_max_length):
    tokens_index = []
    span_end_position = -1
    vocab = tokenizer.vocab
    if whole_word_masking:
        src_wwm = []
        src_length = len(src)
        has_cls, has_sep = False, False
        if src[0] == vocab.get(CLS_TOKEN):
            src = src[1:]
            has_cls = True
        if src[-1] == vocab.get(SEP_TOKEN):
            src = src[:-1]
            has_sep = True
        sentence = "".join(tokenizer.convert_ids_to_tokens(src)).replace('[UNK]', '').replace('##', '')
        import jieba
        wordlist = jieba.cut(sentence)
        if has_cls:
            src_wwm += [vocab.get(CLS_TOKEN)]
        for word in wordlist:
            position = len(src_wwm)
            src_wwm += tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))
            if len(src_wwm) < src_length:
                tokens_index.append([position, len(src_wwm)-position])
        if has_sep:
            src_wwm += [vocab.get(SEP_TOKEN)]
        if len(src_wwm) > src_length:
            src = src_wwm[:src_length]
        else:
            src = src_wwm
    else:
        for (i, token) in enumerate(src):
            if token == vocab.get(CLS_TOKEN) or token == vocab.get(SEP_TOKEN) or token == PAD_ID:
                continue
            if not span_masking:
                tokens_index.append([i])
            else:
                if i < span_end_position:
                    continue
                span_len = get_span_len(span_max_length, span_geo_prob)
                span_end_position = i + span_len
                if span_end_position > len(src):
                    span_len = len(src) - i
                tokens_index.append([i, span_len])
    return tokens_index, src


def get_span_len(max_span_len, p):
    geo_prob_cum = [0.0]
    geo_prob = 1.0
    for i in range(max_span_len + 1):
        if i == 0:
            continue
        if i == 1:
            geo_prob *= p
            geo_prob_cum.append(geo_prob_cum[-1] + geo_prob)
        else:
            geo_prob *= (1 - p)
            geo_prob_cum.append(geo_prob_cum[-1] + geo_prob)

    prob = geo_prob_cum[-1] * random.random()
    for i in range(len(geo_prob_cum) - 1):
        if prob >= geo_prob_cum[i] and prob < geo_prob_cum[i + 1]:
            current_span_len = i + 1
    return current_span_len


def merge_dataset(dataset_path, workers_num):
    # Merge datasets.
    dataset_writer = open(dataset_path, "wb")
    for i in range(workers_num):
        tmp_dataset_reader = open("dataset-tmp-" + str(i) + ".pt", "rb")
        while True:
            tmp_data = tmp_dataset_reader.read(2^20)
            if tmp_data:
                dataset_writer.write(tmp_data)
            else:
                break
        tmp_dataset_reader.close()
        os.remove("dataset-tmp-" + str(i) + ".pt")
    dataset_writer.close()


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
    """ truncate sequence pair to specific length """
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b

        if random.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()


class Dataset(object):
    def __init__(self, args, vocab, tokenizer):
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.corpus_path = args.corpus_path
        self.dataset_path = args.dataset_path
        self.seq_length = args.seq_length
        self.seed = args.seed
        self.dynamic_masking = args.dynamic_masking
        self.whole_word_masking = args.whole_word_masking
        self.span_masking = args.span_masking
        self.span_geo_prob = args.span_geo_prob
        self.span_max_length = args.span_max_length
        self.docs_buffer_size = args.docs_buffer_size
        self.dup_factor = args.dup_factor

    def build_and_save(self, workers_num):
        """
        Build dataset from the given corpus.
        Start workers_num processes and each process deals with a part of data.
        """
        lines_num = count_lines(self.corpus_path)
        print("Starting %d workers for building datasets ... " % workers_num)
        assert (workers_num >= 1)
        if workers_num == 1:
            self.worker(0, 0, lines_num)
        else:
            pool = Pool(workers_num)
            for i in range(workers_num):
                start = i * lines_num // workers_num
                end = (i + 1) * lines_num // workers_num
                pool.apply_async(func=self.worker, args=[i, start, end])
            pool.close()
            pool.join()

        # Merge datasets.
        merge_dataset(self.dataset_path, workers_num)

    def worker(self, proc_id, start, end):
        raise NotImplementedError()


class DataLoader(object):
    def __init__(self, args, dataset_path, batch_size, proc_id, proc_num, shuffle=False):
        self.tokenizer = args.tokenizer
        self.batch_size = batch_size
        self.instances_buffer_size = args.instances_buffer_size
        self.proc_id = proc_id
        self.proc_num = proc_num
        self.shuffle = shuffle
        self.dataset_reader = open(dataset_path, "rb")
        self.read_count = 0
        self.start = 0
        self.end = 0
        self.buffer = []
        self.vocab = args.vocab
        self.whole_word_masking = args.whole_word_masking
        self.span_masking = args.span_masking
        self.span_geo_prob = args.span_geo_prob
        self.span_max_length = args.span_max_length

    def _fill_buf(self):
        try:
            self.buffer = []
            while True:
                instance = pickle.load(self.dataset_reader)
                self.read_count += 1
                if (self.read_count - 1) % self.proc_num == self.proc_id:
                    self.buffer.append(instance)
                    if len(self.buffer) >= self.instances_buffer_size:
                        break
        except EOFError:
            # Reach file end.
            self.dataset_reader.seek(0)

        if self.shuffle:
            random.shuffle(self.buffer)
        self.start = 0
        self.end = len(self.buffer)

    def _empty(self):
        return self.start >= self.end

    def __del__(self):
        self.dataset_reader.close()


class BertDataset(Dataset):
    """
    Construct dataset for MLM and NSP tasks from the given corpus.
    Each document consists of multiple sentences,
    and each sentence occupies a single line.
    Documents in corpus must be separated by empty lines.
    """

    def __init__(self, args, vocab, tokenizer):
        super(BertDataset, self).__init__(args, vocab, tokenizer)
        self.short_seq_prob = args.short_seq_prob

    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        docs_buffer = []
        document = []
        pos = 0
        dataset_writer = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                f.readline()
                pos += 1
            while True:
                line = f.readline()
                pos += 1

                if pos >= end:
                    if len(docs_buffer) > 0:
                        instances = self.build_instances(docs_buffer)
                        for instance in instances:
                            pickle.dump(instance, dataset_writer)
                    break

                if not line.strip():
                    if len(document) >= 1:
                        docs_buffer.append(document)
                    document = []
                    if len(docs_buffer) == self.docs_buffer_size:
                        # Build instances from documents.
                        instances = self.build_instances(docs_buffer)
                        # Save instances.
                        for instance in instances:
                            pickle.dump(instance, dataset_writer)
                        # Clear buffer.
                        docs_buffer = []
                    continue
                sentence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(line))
                if len(sentence) > 0:
                    document.append(sentence)

        dataset_writer.close()

    def build_instances(self, all_documents):
        instances = []
        for _ in range(self.dup_factor):
            for doc_index in range(len(all_documents)):
                instances.extend(self.create_ins_from_doc(all_documents, doc_index))
        return instances

    def create_ins_from_doc(self, all_documents, document_index):
        document = all_documents[document_index]
        max_num_tokens = self.seq_length - 3
        target_seq_length = max_num_tokens
        if random.random() < self.short_seq_prob:
            target_seq_length = random.randint(2, max_num_tokens)
        instances = []
        current_chunk = []
        current_length = 0
        i = 0
        while i < len(document):
            segment = document[i]
            current_chunk.append(segment)
            current_length += len(segment)
            if i == len(document) - 1 or current_length >= target_seq_length:
                if current_chunk:
                    a_end = 1
                    if len(current_chunk) >= 2:
                        a_end = random.randint(1, len(current_chunk) - 1)

                    tokens_a = []
                    for j in range(a_end):
                        tokens_a.extend(current_chunk[j])

                    tokens_b = []
                    is_random_next = 0

                    if len(current_chunk) == 1 or random.random() < 0.5:
                        is_random_next = 1
                        target_b_length = target_seq_length - len(tokens_a)

                        for _ in range(10):
                            random_document_index = random.randint(0, len(all_documents) - 1)
                            if random_document_index != document_index:
                                break

                        random_document = all_documents[random_document_index]
                        random_start = random.randint(0, len(random_document) - 1)
                        for j in range(random_start, len(random_document)):
                            tokens_b.extend(random_document[j])
                            if len(tokens_b) >= target_b_length:
                                break

                        num_unused_segments = len(current_chunk) - a_end
                        i -= num_unused_segments

                    else:
                        is_random_next = 0
                        for j in range(a_end, len(current_chunk)):
                            tokens_b.extend(current_chunk[j])

                    truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)

                    src = []
                    src.append(self.vocab.get(CLS_TOKEN))
                    src.extend(tokens_a)
                    src.append(self.vocab.get(SEP_TOKEN))
                    seg_pos = [len(src)]
                    src.extend(tokens_b)
                    src.append(self.vocab.get(SEP_TOKEN))
                    seg_pos.append(len(src))

                    while len(src) != self.seq_length:
                        src.append(PAD_ID)

                    if not self.dynamic_masking:
                        src, tgt_mlm = mask_seq(src, self.tokenizer, self.whole_word_masking, self.span_masking, self.span_geo_prob, self.span_max_length)
                        instance = (src, tgt_mlm, is_random_next, seg_pos)
                    else:
                        instance = (src, is_random_next, seg_pos)

                    instances.append(instance)
                current_chunk = []
                current_length = 0
            i += 1
        return instances


class BertDataLoader(DataLoader):
    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            src = []
            tgt_mlm = []
