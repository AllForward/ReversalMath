import pprint
import numpy as np
import torch

def last_boxed_only(sample):
    q, a = sample
    a = last_boxed_only_string(a)
    if a == None:
        return None
    return (q, a)

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
    
    return retval

def only_until_first_boxed_from_tokens(string, tokens):
    idx = string.find("\\boxed")
    if idx < 0:
        idx = string.find("\\fbox")
        if idx < 0:
            return None
    
    cum_length = 0
    for i, t in enumerate(tokens):
        cum_length += len(t)
        if cum_length >= idx:
            break
    
    return tokens[:i]



def clean_numbers(sample):
    if not sample:
        return None
    new_sample = list()
    for s in sample:
        new_sample.append(_clean_numbers(s))

    return tuple(new_sample)

def _clean_numbers(string):
    """
    Clean Numbers in the given string

    >>> _clean_numbers(None, "Hello 123")
    'Hello 123'
    >>> _clean_numbers(None, "Hello 1234")
    'Hello 1,234'
    >>> _clean_numbers(None, "Hello 1234324asdasd")
    'Hello 1,234,324asdasd'
    """
    num_prev_digits = 0
    new_string = ""
    for i, c in enumerate(string):
        # isdigit() doesnt work here because of weird unicode chars.
        if c in {'1', '2', '3', '4', '5', '6', '7', '8', '9', '0'}:
            num_prev_digits += 1
        else:
            if num_prev_digits > 3:
                # Some fixing
                string_number = new_string[-num_prev_digits:]
                new_string = new_string[:-num_prev_digits] + "{0:,}".format(int(string_number))
            num_prev_digits = 0
        new_string += c

    if num_prev_digits > 3:
        # Some fixing
        string_number = new_string[-num_prev_digits:]
        new_string = new_string[:-num_prev_digits] + "{0:,}".format(int(string_number))

    return new_string

def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string

def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string

def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string


def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        #pdb.set_trace()
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2

class NotEqual:
    def __eq__(self, other):
        return False
    
def update_decoder_input_and_label(input_ids, sources_tokenized, tokenizer):
    # 对 decoder_input 进行 random mask
    train_llm_rationales = input_ids
    update_decoder_input = []
    update_decoder_label = []
    decoder_attention_mask = []
    max_len = 0
    pos_len_list = []

    for label, source_len in zip(train_llm_rationales, sources_tokenized):
        label = label[label != tokenizer.pad_token_id]
        label = label[label != tokenizer.eos_token_id]
        question_len = source_len - 1
        qs_explanation = label.numpy().copy()
        label = qs_explanation[question_len + 1:]
        # find the step based on the '.'
        symbol_index = np.where(label == 29889)[0]
        # remove the float number split by '.'
        filt_index = []
        for i in range(len(symbol_index)):
            if tokenizer.decode(label[symbol_index[i] - 1]).isdigit() and symbol_index[i] < len(label) - 1 \
                and tokenizer.decode(label[symbol_index[i] + 1]).isdigit():
                filt_index.append(i)
        symbol_index = np.delete(symbol_index, filt_index)
        pos_len = {}
        decoder_input = label.tolist()
        # decoder_label = label.tolist()
        if len(symbol_index) == 0:
            # decoder_input.insert(0, tokenizer.bos_token_id)
            decoder_input.append(tokenizer.eos_token_id)
            # decoder_label.append(tokenizer.eos_token_id)
            pos_len[question_len + 1] = len(label)
        else:
            if symbol_index[-1] != len(label) - 1:
                step_len = len(symbol_index) + 1
            else:
                step_len = len(symbol_index)
            random_arry = np.random.rand(step_len) < np.random.rand()
            while random_arry.sum() < 1 and step_len > 0:
                random_arry = np.random.rand(step_len) < np.random.rand()
            add_num = 0
            start_index = 0 if np.where(random_arry)[0][0] == 0 else symbol_index[np.where(random_arry)[0][0] - 1] + 1
            end_index = 0
            for i in range(step_len):
                if random_arry[i]:
                    if i < step_len - 1:
                        decoder_input.insert(symbol_index[i] + add_num + 1, tokenizer.eos_token_id)
                        add_num += 1
                        # decoder_input.insert(start_index, tokenizer.bos_token_id)
                        # add_num += 1
                        if i == 0:
                            # if len(symbol_index) == 1:
                            pos_len[question_len + 1 + start_index] = symbol_index[i] + 1
                        else:
                            pos_len[question_len + 1 + start_index] = symbol_index[i] - symbol_index[i-1]
                        end_index = symbol_index[i] + add_num
                        start_index = end_index + 1
                    else:
                        if step_len == len(symbol_index):
                            if i > 0:
                                # decoder_input.insert(symbol_index[-2] + add_num + 1, tokenizer.bos_token_id)
                                decoder_input.append(tokenizer.eos_token_id)
                                pos_len[question_len + 1 + symbol_index[-2] + add_num + 1] = symbol_index[i] - symbol_index[i-1]
                            else:
                                # decoder_input.insert(0, tokenizer.bos_token_id)
                                decoder_input.append(tokenizer.eos_token_id)
                                pos_len[question_len + 1] = symbol_index[i] + 1
                        else:
                            # decoder_input.insert(symbol_index[-1] + add_num + 1, tokenizer.bos_token_id)
                            decoder_input.append(tokenizer.eos_token_id)
                            pos_len[question_len + 1 + symbol_index[-1] + add_num + 1] = len(label) - 1 - symbol_index[-1]
                else:
                    start_index = symbol_index[i] + add_num + 1 if i < (step_len - 1) else -1
        decoder_input = np.append(qs_explanation[:question_len + 1], decoder_input)
        if decoder_input[-1] != tokenizer.eos_token_id:
            decoder_input = np.append(decoder_input, tokenizer.eos_token_id)
        max_len = max(len(decoder_input), max_len)
        update_decoder_input.append(decoder_input)
        update_decoder_label.append(decoder_input.copy())
        pos_len_list.append(pos_len)

    # 对 input 和 label 统一长度
    un_label_pos = torch.zeros([len(update_decoder_input), max_len])
    for i in range(len(update_decoder_input)):
        ori_decoder_input_len = len(update_decoder_input[i])
        if len(update_decoder_input[i]) < max_len:
            # decoder only uses left padding
            # update_decoder_input[i] = np.append([tokenizer.pad_token_id] * (max_len - len(update_decoder_input[i])), update_decoder_input[i])
            # update_decoder_label[i] = np.append([-100] * (max_len - len(update_decoder_label[i])), update_decoder_label[i])

            # decoder only uses right padding
            update_decoder_input[i] = np.append(update_decoder_input[i], [tokenizer.pad_token_id] * (max_len - len(update_decoder_input[i])))
            update_decoder_label[i] = np.append(update_decoder_label[i], [-100] * (max_len - len(update_decoder_label[i])))
        # sep_pos = np.where(np.array(update_decoder_input[i]) == tokenizer.sep_token_id)[0].item()
        sep_pos = sources_tokenized[i] - 1
        update_decoder_label[i][:sep_pos + 1] = -100
        
        # 获取 attention mask matrix
        attention_mask = np.ones([max_len, max_len])
        pos_len = pos_len_list[i]
        pos_len_key = list(pos_len.keys())
        
        # question 部分为下三角
        attention_mask[:sep_pos+1, :sep_pos+1] = np.tril(np.ones([sep_pos+1, sep_pos+1]), 0)
        attention_mask[:sep_pos+1, sep_pos+1:] = 0.0

        for j in range(len(pos_len_key)):
            # left padding
            # key, val = pos_len_key[j] + sep_pos + 1, pos_len[pos_len_key[j]]
            # right padding
            key, val = pos_len_key[j], pos_len[pos_len_key[j]]

            # un_label_pos[i][key:key+val+2] = 1
            # due to Shift right auto in gpt-2
            # un_label_pos[i][key+1:key+val+2] = 1
            # un_label_pos[i][key+1:key+val+1] = 1
            un_label_pos[i][key:key+val+1] = 1
            
            # attention_mask[:, key:key+val+2] = 0.0
            attention_mask[:, key:key+val+1] = 0.0
            try:
                # attention_mask[key:key+val+2, key:key+val+2] = np.tril(np.ones([val+2, val+2]), 0)
                attention_mask[key:key+val+1, key:key+val+1] = np.tril(np.ones([val+1, val+1]), 0)
                if j > 0:
                    # import pdb; pdb.set_trace()
                    pre_num = j
                    while pre_num > 0:
                        # pre_key, pre_val = pos_len_key[pre_num-1] + sep_pos + 1, pos_len[pos_len_key[pre_num-1]]
                        pre_key, pre_val = pos_len_key[pre_num-1], pos_len[pos_len_key[pre_num-1]]
                        # attention_mask[key:key+val+2, pre_key:pre_key+pre_val+2] = 1.0
                        attention_mask[key:key+val+1, pre_key:pre_key+pre_val+1] = 1.0
                        pre_num -= 1
            except:
                import pdb; pdb.set_trace()
                print()
        if ori_decoder_input_len < max_len:
            # attention_mask[:, :max_len - ori_decoder_input_len] = 0.0
            # attention_mask[:max_len - ori_decoder_input_len, :] = 0.0
            attention_mask[:, ori_decoder_input_len:] = 0.0
            attention_mask[ori_decoder_input_len:, :] = 0.0
        decoder_attention_mask.append(attention_mask.tolist())

        update_decoder_input[i] = update_decoder_input[i].tolist()
        update_decoder_label[i] = update_decoder_label[i].tolist()
    # import pdb; pdb.set_trace()
    decoder_attention_mask = torch.FloatTensor(decoder_attention_mask)
    # decoder_attention_mask.masked_fill_(decoder_attention_mask == 0.0, torch.finfo(torch.float32).min)
    # decoder_attention_mask.masked_fill_(decoder_attention_mask == 1.0, 0.0)

    update_decoder_label = torch.LongTensor(update_decoder_label)
    update_decoder_label.masked_fill_(un_label_pos != 1, -100)
    
    return torch.LongTensor(update_decoder_input), update_decoder_label, decoder_attention_mask