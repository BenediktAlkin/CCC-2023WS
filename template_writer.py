lines = [
    "line1",
    "line2",
]
with open("out.csv", "w") as f:
    lines = [f"{str(line)}\n" for line in lines[:-1]] + [str(lines[-1])]
    f.writelines(lines)
