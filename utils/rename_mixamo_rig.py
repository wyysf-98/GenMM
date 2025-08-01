# rename_mixamo_prefix.py
import bpy, re
rx = re.compile(r"mixamorig\d+:")          # any number before the colon

for obj in bpy.data.objects:
    if obj.type == 'ARMATURE':
        for b in obj.data.bones:
            b.name = rx.sub("mixamorig:", b.name)