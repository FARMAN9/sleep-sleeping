import glob

labels_path = "COCO/labels/train2017/*.txt"
classes = set()

for label_file in glob.glob(labels_path):
    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            # skip lines that don't start with a number
            if parts[0].isdigit():
                class_id = int(parts[0])
                classes.add(class_id)

print("Unique class IDs in dataset:", classes)
print("Total number of classes:", len(classes))
