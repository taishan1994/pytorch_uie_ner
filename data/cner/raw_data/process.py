import os
import re
import json

en2ch = {
  'PRO':'专业', 
  'ORG':'机构', 
  'CONT':'国籍', 
  'RACE':'民族', 
  'NAME':'人名', 
  'EDU':'学历', 
  'LOC':'籍贯', 
  'TITLE':'职称',
}

def preprocess(input_path, save_path, mode):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    data_path = os.path.join(save_path, mode + ".json")
    labels = set()
    result = []
    tmp = {}
    tmp['id'] = 0
    tmp['text'] = ''
    tmp['relations'] = []
    tmp['entities'] = []
    # =======先找出句子和句子中的所有实体和类型=======
    with open(input_path,'r',encoding='utf-8') as fp:
        lines = fp.readlines()
        texts = []
        entities = []
        words = []
        entity_tmp = []
        entities_tmp = []
        for line in lines:
            line = line.strip().split(" ")
            if len(line) == 2:
                word = line[0]
                label = line[1]
                words.append(word)

                if "B-" in label:
                    entity_tmp.append(word)
                elif "M-" in label:
                    entity_tmp.append(word)
                elif "E-" in label:
                    entity_tmp.append(word)
                    if ("".join(entity_tmp), label.split("-")[-1]) not in entities_tmp:
                        entities_tmp.append(("".join(entity_tmp), en2ch[label.split("-")[-1]]))
                    labels.add(en2ch[label.split("-")[-1]])
                    entity_tmp = []

                if "S-" in label:
                    entity_tmp.append(word)
                    if ("".join(entity_tmp), label.split("-")[-1]) not in entities_tmp:
                        entities_tmp.append(("".join(entity_tmp), en2ch[label.split("-")[-1]]))
                    entity_tmp = []
                    labels.add(en2ch[label.split("-")[-1]])
            else:
                texts.append("".join(words))
                entities.append(entities_tmp)
                words = []
                entities_tmp = []

        # for text,entity in zip(texts, entities):
        #     print(text, entity)
        # print(labels)
    # ==========================================
    # =======找出句子中实体的位置=======
    i = 0
    for text,entity in zip(texts, entities):

        if entity:
            ltmp = []
            for ent,type in entity:
                for span in re.finditer(ent, text):
                    start = span.start()
                    end = span.end()
                    ltmp.append((type, start, end, ent))
                    # print(ltmp)
            ltmp = sorted(ltmp, key=lambda x:(x[1],x[2]))
            for j in range(len(ltmp)):
                # tmp['entities'].append(["".format(str(j)), ltmp[j][0], ltmp[j][1], ltmp[j][2], ltmp[j][3]])
                tmp['entities'].append({"id":j, "start_offset":ltmp[j][1], "end_offset":ltmp[j][2], "label":ltmp[j][0]})
        else:
            tmp['entities'] = []
        tmp['id'] = i
        tmp['text'] = text
        result.append(tmp)
        tmp = {}
        tmp['id'] = 0
        tmp['text'] = ''
        tmp['relations'] = []
        tmp['entities'] = []
        i += 1

    with open(data_path, 'w', encoding='utf-8') as fp:
        fp.write("\n".join([json.dumps(i, ensure_ascii=False) for i in result]))


preprocess("train.char.bmes", '../mid_data', "train")
preprocess("dev.char.bmes", '../mid_data', "dev")
preprocess("test.char.bmes", '../mid_data', "test")

