# Credits to this answer https://stackoverflow.com/a/52837006/8673150
# list of cjk codepoint ranges
# tuples indicate the bottom and top of the range, inclusive
cjk_ranges = [(0x4E00, 0x62FF), (0x6300, 0x77FF), (0x7800, 0x8CFF),
              (0x8D00, 0x9FCC), (0x3400, 0x4DB5), (0x20000, 0x215FF),
              (0x21600, 0x230FF), (0x23100, 0x245FF), (0x24600, 0x260FF),
              (0x26100, 0x275FF), (0x27600, 0x290FF), (0x29100, 0x2A6DF),
              (0x2A700, 0x2B734), (0x2B740, 0x2B81D), (0x2B820, 0x2CEAF),
              (0x2CEB0, 0x2EBEF), (0x2F800, 0x2FA1F)]


def is_cjk(char):
    char = ord(char)
    for bottom, top in cjk_ranges:
        if bottom <= char <= top:
            return True
    return False


def is_number(char):
    # Determine whether a character is a number.
    char = ord(char)
    if 0x0030 <= char <= 0x0039:
        return True
    else:
        return False


def is_alphabet(char):
    # Determine whether a character is an English letter.
    char = ord(char)
    if (0x0041 <= char <= 0x005a) or (0x0061 <= char <= 0x007a):
        return True
    else:
        return False


# Reference: https://www.w3schools.com/charsets/ref_utf_greek.asp
def is_greek(char):
    # Determine whether a character is a Greek letter.
    char = ord(char)
    if 0x0370 <= char <= 0x03FF:
        return True
    else:
        return False


if __name__ == "__main__":
    ch = 'α'
    print(is_cjk(ch))
    print(is_number(ch))
    print(is_alphabet(ch))
    print(is_greek(ch))
    for code_point in range(880, 1024):
        print(chr(code_point))
