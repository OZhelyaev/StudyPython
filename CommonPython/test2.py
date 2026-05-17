s = input()
f1 = s.find('h')
f2 = s.rfind('h')

print(s[:f1] + s[f2 + 1:])

