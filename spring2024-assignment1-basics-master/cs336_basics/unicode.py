test_string = "hello! こんにちは!"

utf8_encoded = test_string.encode("utf-8")
utf16_encoded = test_string.encode("utf-16")
utf32_encoded = test_string.encode("utf-32")

print(utf8_encoded)
print(utf16_encoded)
print(utf32_encoded)
print(list(utf8_encoded))
print(list(utf16_encoded))
print(list(utf32_encoded))
