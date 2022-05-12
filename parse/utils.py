def recognize_diff(input_s):
    '''
    extract the first expression of diff, not parse the inner diff


    :param input_s:
    :return:
    '''
    result = []

    loc1 = 0 #记录初始位置
    loc2 = 0 #记录第一层diff内容的终止位置
    diff_count =input_s.count('diff')
    for i, s in enumerate(input_s):
        if s==',' and diff_count==0:
            loc2 = i
            break
            # diff_count+=0
        elif s==',' and diff_count!=0:
            diff_count-=1
    result.append(input_s[loc1:loc2])
    result.append(input_s[loc2+1:])
    return result

def match(current_token, token):
    if current_token.token_type == token:
        return current_token
    else:
        raise Exception('Invalid syntax on token {}'.format(current_token.token_type))


if __name__ == '__main__':
    print(recognize_diff('f * k * diff(u,x),u'))