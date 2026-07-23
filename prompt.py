KEEP_TAIL = (
    "keeping image 1's identity, facial geometry, skin tone/undertone, hair, pose, "
    "lighting, composition, and background unchanged—no face replacement."
)

ALIASES = {"eye":"eyes","mouth":"lip","lips":"lip","blush":"face","stickers":"face","face_accents":"face"}

def _parse_labels(label_input):
    # 支持字符串和列表
    if isinstance(label_input, str):
        labels = [lab.strip() for lab in label_input.split(',') if lab.strip()]
    elif isinstance(label_input, list):
        labels = [str(x).strip() for x in label_input if str(x).strip()]
    else:
        labels = [str(label_input).strip()]
    return labels

def _norm_labels(labels):
    norm = []
    for lab in labels:
        k = ALIASES.get(lab.lower(), lab.lower())
        norm.append(k)
    # 去重并排序
    picked = []
    for k in ["eyes","lip","face"]:
        if k in norm and k not in picked:
            picked.append(k)
    return picked

# 三个固定 prompt
PROMPT1 = f"Apply eye shadow, eyeliner, lashes, brows from image 2 to image 1. Keeping image 1's identity, facial features, lip, cheek, complexion, hair, pose, lighting, composition, and background unchanged,no face replacement"
PROMPT2 = f"Apply lipstick from image 2 to image 1. Keeping image 1's identity, facial features, eyes, cheek, complexion, hair, pose, lighting, composition, and background unchanged,no face replacement."
PROMPT3 = f"Apply eye shadow, eyeliner, lashes, brows, lipstick, blush, cheek painting from image 2 to image 1. Keeping image 1's identity, facial features, complexion, hair, pose, lighting, composition, and background unchanged,no face replacement."
PROMPT_DEFAULT = PROMPT3                                            

ref1_localization_prompt1="identity, facial features, lip, cheek, complexion"
ref2_localization_prompt1="eyeshadow, eyeliner, lashes, brows" 
caption_ref1_location1=[26, 28, 29, 31, 33, 35]
caption_ref2_location1=[1, 2, 4, 5, 6, 9, 12, 13]

ref1_localization_prompt2="identity, facial features, eyes, cheek, complexion"
ref2_localization_prompt2="lipstick"
caption_ref1_location2=[14, 16, 17, 19, 21, 23]
caption_ref2_location2=[1]

ref1_localization_prompt3="identity, facial features, complexion"
ref2_localization_prompt3="eyeshadow ,eyeliner, lashes, brows, lipstick"
caption_ref1_location3=[33, 35, 36, 38]
caption_ref2_location3=[1, 2, 4, 5, 6, 9, 12, 13, 15]

def build_prompt(label_input):
    labels = _norm_labels(_parse_labels(label_input))
    s = set(labels)
    if s == {"eyes"}:
        return PROMPT1,ref1_localization_prompt1,ref2_localization_prompt1,caption_ref1_location1,caption_ref2_location1
    if s == {"lip"}:
        return PROMPT2,ref1_localization_prompt2,ref2_localization_prompt2,caption_ref1_location2,caption_ref2_location2
    if s == {"eyes","lip","face"}:
        return PROMPT3,ref1_localization_prompt3,ref2_localization_prompt3,caption_ref1_location3,caption_ref2_location3
    return PROMPT_DEFAULT,caption_ref1_location3,caption_ref2_location3

# 简单测试
if __name__ == "__main__":
    print(build_prompt(['eyes']))               # prompt1
    print(build_prompt(['lip']))                # prompt2
    print(build_prompt(['eyes','lip','face']))  # prompt3
    print(build_prompt('eye'))                  # 别名 -> eyes -> prompt1
    print(build_prompt('unknown'))              # 默认
