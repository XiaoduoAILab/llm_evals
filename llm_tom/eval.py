import logging
import os
import random
import warnings

import numpy as np
import torch
import transformers
from tqdm import tqdm


@torch.no_grad()
def generate(prompt, temperature=.0001, max_tokens=50):
    inputs = tokenizer.encode(prompt, padding=False, return_tensors="pt").to(model.device)
    num_tokens = inputs.shape[-1]
    outputs = model.generate(inputs, max_new_tokens=max_tokens, pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(outputs[0][num_tokens:]).strip()

    log(text)

    text = text.replace(',', ' ').replace('.', ' ').replace("'", ' ')
    text = text.split(' ')[0]
    out = text
    return out


## Function to process the tasks
def process_tsk(tsk, len_out=1, study="S1"):
    if study == "S1":
        txt = tsk["txt"].replace("S1", tsk["o1"]).replace("S2", tsk["o2"]).replace("CX", tsk["c"]).replace("XNAM", tsk[
            "xnam"]).replace("XPRO", tsk["xpro"]).replace("POS_PRO", tsk["pos_pro"]).replace(". h", ". H").replace(
            ". s", ". S")
        txtr = txt.replace(tsk["o1"], "X1").replace(tsk["o2"], tsk["o1"]).replace("X1", tsk["o2"])

        q1 = tsk["xpro"].capitalize() + " opens the " + tsk["c"] + " and looks inside. " + tsk[
            "xpro"].capitalize() + " can clearly see that it is full of"
        q2 = tsk["xpro"].capitalize() + " believes that the " + tsk["c"] + " is full of"
        q3 = tsk["xpro"].capitalize() + " calls " + tsk["pos_pro"] + " friend to tell them that " + tsk[
            "xpro"] + " has just found a " + tsk["c"] + " full of"
        # correct task #17
        if tsk["NUMBER"] == 17:
            q1 = tsk["xpro"].capitalize() + " puts the " + tsk["c"] + " on and listens to it. " + tsk[
                "xpro"].capitalize() + " can clearly hear that it is full of"

    if study == "S2":
        txt = tsk["txt"]
        txtr = txt.replace(tsk["o1"], "PLACEHOLDER_FOR_O1").replace(tsk["o2"], tsk["o1"]).replace("PLACEHOLDER_FOR_O1",
                                                                                                  tsk["o2"])
        q1 = tsk["q1"]
        q2 = tsk["q2"]
        q3 = tsk["q3"]

    log("================================= TASK #" + str(tsk["NUMBER"]) + " ================================= ")
    log("TASK: " + txt)
    log("REVERSED TASK: " + txtr)
    log("PROMPT 1.1: " + q1 + " ______")
    log("PROMPT 1.2: " + q2 + " ______")
    log("PROMPT 1.3: " + q3 + " ______")
    log("=========================================================================== ")

    ### normal
    log("## EXPECTED PATTERN: " + tsk["o1"] + "--" + tsk["o2"] + "--" + tsk["o2"])

    gen1 = generate(prompt=txt + q1, max_tokens=len_out, temperature=0)
    log(gen1)
    gen2 = generate(prompt=txt + q2, max_tokens=len_out, temperature=0)
    log(gen2)
    gen3 = generate(prompt=txt + q3, max_tokens=len_out, temperature=0)
    log(gen3)

    score = (gen1 in tsk["o1"]) and (gen2 in tsk["o2"]) and (gen3 in tsk["o2"])
    if args.debug:
        print('gen1: ' + str(gen1))
        print('tsk["o1"]: ' + str(tsk["o1"]))
        print('gen1 in tsk["o1"]: ' + str(gen1 in tsk["o1"]))

        print('gen2: ' + str(gen2))
        print('tsk["o2"]: ' + str(tsk["o2"]))
        print('gen2 in tsk["o2"]: ' + str(gen2 in tsk["o2"]))

        print('gen3: ' + str(gen3))
        print('tsk["o2"]: ' + str(tsk["o2"]))
        print('gen3 in tsk["o2"]: ' + str(gen3 in tsk["o2"]))

    ### reversed
    log("## EXPECTED PATTERN: " + tsk["o2"] + "--" + tsk["o1"] + "--" + tsk["o1"])

    gen1 = generate(prompt=txtr + q1, max_tokens=len_out, temperature=0)
    log(gen1)
    gen2 = generate(prompt=txtr + q2, max_tokens=len_out, temperature=0)
    log(gen2)
    gen3 = generate(prompt=txtr + q3, max_tokens=len_out, temperature=0)
    log(gen3)

    score = score and (gen1 in tsk["o2"]) and (gen2 in tsk["o1"]) and (gen3 in tsk["o1"])
    if args.debug:
        print('gen1: ' + str(gen1))
        print('tsk["o2"]: ' + str(tsk["o2"]))
        print('gen1 in tsk["o2"]: ' + str(gen1 in tsk["o2"]))

        print('gen2: ' + str(gen2))
        print('tsk["o1"]: ' + str(tsk["o1"]))
        print('gen2 in tsk["o1"]: ' + str(gen2 in tsk["o1"]))

        print('gen3: ' + str(gen3))
        print('tsk["o1"]: ' + str(tsk["o1"]))
        print('gen3 in tsk["o1"]: ' + str(gen3 in tsk["o1"]))

    score = int(score)
    return score


def get_logger(logname):
    if os.path.exists(logname):
        os.remove(logname)

    logging.basicConfig(filename=logname,
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)

    logger = logging.getLogger()
    return logger


def seed_everything(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    seed_everything(42)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To avoid warnings about parallelism in tokenizers
    warnings.filterwarnings("ignore")

    import argparse

    parser = argparse.ArgumentParser(description='Theory of Mind')
    # general
    parser.add_argument('--model', type=str, default='/mnt/sdb/ly/models/hf_converted_llama/7B/', help='')
    parser.add_argument('--debug', type=bool, default=False, help='debug')

    args = parser.parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, skip_special_tokens=False, add_bos_token=False,
                                                           add_eos_token=False, clean_up_tokenization_spaces=True,
                                                           use_fast=False,
                                                           trust_remote_code=True)

    print(f"Loading model from {args.model}")
    model_config = transformers.AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    print(model_config)

    torch_dtype = model_config.torch_dtype
    if torch_dtype == torch.float32 or torch_dtype == torch.float64:
        torch_dtype = torch.float16

    model = transformers.AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch_dtype, device_map='auto',
                                                              trust_remote_code=True)

    model.eval()

    print(model)
    print(tokenizer)

    model_name = os.path.basename(os.path.dirname(args.model))
    logname = '{}.txt'.format(model_name)
    logger = get_logger(logname)


    def log(msg):
        logger.info(msg)


    #### Sanity Check
    # print(hf("I can query huggingface whenever", model, tokenizer, temperature=1))
    # print(hf("I can query huggingface whenever", model, tokenizer, temperature=1))
    # print(hf("I can query huggingface whenever", model, tokenizer, temperature=1))

    ### Study 1
    txt1 = r"""Here is a bag filled with popcorn. There is no chocolate in the bag. Yet, the label on the bag says 'chocolate' and not 'popcorn.' Sam finds the bag. She had never seen the bag before. She cannot see what is inside the bag. She reads the label. """

    ### Prompt 1.1: Contents
    qcontents = "She opens the bag and inspects its contents. She can clearly see that it is full of"
    log(generate(prompt=txt1 + qcontents, max_tokens=90))

    #### Prompt 1.2: Protagonist's belief
    qbelief = "She believes that the bag is full of"
    log(generate(prompt=txt1 + qbelief, max_tokens=50, temperature=0))

    #### Prompt 1.3: Protagonist's belief
    qbelief = "She is delighted to have found this bag. She loves eating"
    log(generate(prompt=txt1 + qbelief, max_tokens=50, temperature=0))

    ### Study 2
    txt = r"""In the room, there are John, Mark, a cat, a box, and a basket. 
    John takes the cat and puts it in the basket. 
    He leaves the room and goes to school. 
    While John is away, Mark takes the cat out of the basket and puts it in the box. 
    Mark leaves the room and goes to work. John comes back from school and enters the room. 
    He doesn't know what happened in the room when he was away.
    """.replace("\n", " ").replace("  ", " ")

    ### Prompt 1.1: Location
    location = "The cat jumps out of the"
    log(generate(prompt=txt + location, max_tokens=90))

    ### Prompt 1.2: Protagonist's belief
    qbelief = "John thinks that the cat is in the"

    log(generate(prompt=txt + qbelief, max_tokens=50, temperature=0))

    ### Prompt 1.3: Protagonist's belief
    qbelief = "When John comes back home, he will look for the cat in the"

    log(generate(prompt=txt + qbelief, max_tokens=50, temperature=0))

    ### Study 3
    # unexpected Contents tasks
    tsks1 = [
        {
            "txt": r"""On the shelf, there is a CX. It is full of S1; there is no S2 in it. Yet, the label on this CX says "S2" and not "S1". XNAM walks into the room and notices the CX. XPRO has never seen it before. He reads the label. """,
            "o1": "beer", "o2": "wine", "c": "bottle", "xnam": "Mark", "xpro": "he", "pos_pro": "his",
            "q3": "loves drinking"},

        {
            "txt": r"""On the shelf in the company's headquarters, there is a CX that contains only S1 files and no S2 files. Yet, confusingly, its label clearly states "S2 files" and not "S1 files". The newly hired computer engineer finds the CX on the shelf. XPRO has never seen this CX before. XPRO reads its label. """,
            "o1": "audio", "o2": "video", "c": "hard drive", "xnam": "computer engineer", "xpro": "she",
            "pos_pro": "her", "q3": "loves browsing through old files containing"},

        {
            "txt": r"""In the freezer, there is a CX filled with S1. There is no S2 in it. Yet, the label says "S2" and not "S1". The label is wrong. One day, XNAM finds the CX and realizes that XPRO has never seen it before. XPRO cannot see what is inside the CX. XPRO reads the label. """,
            "o1": "ice cream", "o2": "jam", "c": "container", "xnam": "Anna", "xpro": "she", "pos_pro": "her",
            "q3": "loves eating"},

        {
            "txt": r"""A CX full of S1 arrives at Amsterdam port. There is no S2 in it. It is sealed, and one cannot see what is inside. The shipping documents state that it contains S2. A customs officer approaches the CX. XPRO cannot see what is inside. XPRO reads the shipping documents to figure out what is inside. """,
            "o1": "salt", "o2": "sugar", "c": "shipping container", "xnam": "customs officer", "xpro": "she",
            "pos_pro": "her", "q3": "thinks that it is much easier to inspect shipping containers filled with"},

        {
            "txt": r"""There is a closed CX on the table of an empty meeting room, in which a seminar is about to take place. The CX is filled with S1, and there are no S2 in it. Yet, it is labeled with the word "S2". The first attendee walks into the room and sits in a chair in front of the CX. XPRO reads the label. """,
            "o1": "pens", "o2": "pencils", "c": "box", "xnam": "attendee", "xpro": "he", "pos_pro": "his",
            "q3": "loves writing with"},

        {
            "txt": r"""Inside the fridge of the Airbnb host's kitchen is a CX labeled as "S2". However, the CX actually contains S1 and no S2. XNAM, a guest who just arrived at the house, opens the fridge and sees the CX. XPRO cannot see what is inside the CX, but XPRO reads the label. """,
            "o1": "sardines", "o2": "tuna", "c": "can", "xnam": "Jill", "xpro": "she", "pos_pro": "her",
            "q3": "loves eating"},

        {
            "txt": r"""There is a CX lying in front of Julia's door. The envelope is filled with S1, and there are no S2 in it. Yet, a label with the word "S2" is stuck on the outside. XNAM who's leaving POS_PRO own apartment for the first time that morning, walks past and notices the CX on the floor. XPRO wonders what is inside the CX and reads the label. """,
            "o1": "leaflets", "o2": "receipts", "c": "paper envelope", "xnam": "Julia's neighbor", "xpro": "she",
            "pos_pro": "her", "q3": "hoping that the tenant finally receives some of those"},

        {
            "txt": r"""A vault that's been unopened for over 100 years has just been unlocked. Inside lies a closed CX with an engraving on it that reads "S2". However, the CX actually contains only S1 and doesn't contain any S2 at all. An inspector approaches the vault. XPRO cannot see what is inside the CX. XPRO reads the engraving. """,
            "o1": "gems", "o2": "books", "c": "box", "xnam": "inspector", "xpro": "he", "pos_pro": "his",
            "q3": "was hoping, his whole career, to find such a box filled with"},

        {
            "txt": r"""A CX filled with S1 sits on the counter of a department store. There is no S2 inside it; however, the CX bears the label "S2" instead of "S1". A new customer walks up to the counter and notices the bottle. He reads the label. """,
            "o1": "perfume", "o2": "water", "c": "bottle", "xnam": "new customer", "xpro": "he", "pos_pro": "his",
            "q3": "was just looking for a bottle full of"},

        {
            "txt": r"""In an abandoned mansion, a forensics team is inspecting a crime scene. Next to the body is a CX. It contains S1, but it remains closed, and no one can see what is inside. The label on this CX says S2. The XNAM, who's just arrived on the scene, crouches down to inspect the CX. He cannot see what is inside the CX, but he reads the label. """,
            "o1": "cocaine", "o2": "heroin", "c": "metal box", "xnam": "lead detective", "xpro": "he", "pos_pro": "his",
            "q3": "specializes in cases which involve metal cans filled with drugs such as"},

        {
            "txt": r"""Julia prepares a gift box for her son, XNAM, under the Christmas tree. She wants to prank Steven. She fills the gift box with S1 but labels it S2, written in clear large letters. A few hours later, XNAM comes down the stairs and finds the gift box under the Christmas tree. He cannot see what is inside, but he reads the label. """,
            "o1": "tea", "o2": "coffee", "c": "gift box", "xnam": "Steven", "xpro": "he", "pos_pro": "his",
            "q3": "loves drinking"},

        {
            "txt": r"""At the customs division of an airport, an immigration officer hands over a CX to a biohazard inspector. The CX is sealed and labeled as "S2 samples". No one can see what is inside, but in fact, it only contains S1 samples. The inspector has not examined the contents of this CX before. She reads the label on the CX. """,
            "o1": "blood", "o2": "urine", "c": "plastic box", "xnam": "inspector", "xpro": "she", "pos_pro": "her",
            "q3": "loves analysing samples containing"},

        {
            "txt": r"""Paul holds out a gift-wrapped CX to XNAM and tells POS_PRO that there are S2 in it. In reality, though, the CX contains only S1. XNAM takes the CX from him. """,
            "o1": "socks", "o2": "scarves", "c": "box", "xnam": "Sara", "xpro": "she", "pos_pro": "her",
            "q3": "has just run out of"},

        {
            "txt": r"""A deceased person's laptop contains an encrypted folder labeled "S2". Yet, unbeknownst to everyone, the folder only contains S1. There are no S2 in it. A digital forensics expert has been hired to retrieve and back up the contents of this folder. As XPRO switches on the machine for the first time, the home screen appears. She locates the encrypted folder and reads its label. """,
            "o1": "videos", "o2": "photos", "c": "folder", "xnam": "forensics expert", "xpro": "she", "pos_pro": "her",
            "q3": "has all the tools needed to decrypt files containing"},

        {
            "txt": r"""On Thursday, XNAM orders some S2 on the internet. Unfortunately, there is a mistake at the logistics center. They ship a CX full of S1 but label it as "S2". The CX arrives on Saturday in the morning. The postman leaves the CX at XNAM's front door and rings the doorbell. XNAM opens the door, looks down, and sees the CX. He reads the label. """,
            "o1": "wallets", "o2": "sneakers", "c": "parcel", "xnam": "Daniel", "xpro": "he", "pos_pro": "his",
            "q3": "couldn't wait to receive these"},

        {
            "txt": r"""A CX has been left behind at the park after a school's picnic day event. One of the organizers, who's strolling past the bench on which the CX is sitting, sees it. The CX has only S1 in it, but a sticker on the outside says "S2". The organizer doesn't know what's inside; XPRO reads the sticker. """,
            "o1": "sandwiches", "o2": "beer", "c": "cooler box", "xnam": "organizer", "xpro": "he", "pos_pro": "his",
            "q3": "hoping to find some"},

        {
            "txt": r"""The packers at a freight forwarding company are notified of a CX that just arrived at their headquarters. The bill of lading lists the contents of this CX as S2. One of the packers, XNAM, is sent to collect the CX. No one tells him that the CX actually contains S1 and there are no S2 in it. """,
            "o1": "clothes", "o2": "vegetables", "c": "container", "xnam": "Daniel", "xpro": "he", "pos_pro": "his",
            "q3": "hoping to find a container full of"},

        {
            "txt": r"""XNAM is searching for something in POS_PRO mother's attic. XPRO finds a box of CDs labeled "S2". XNAM has never seen or listened to these CDs before, and doesn't know that they contain only S1 music and no S2 music at all. She reads the description written on the box. """,
            "o1": "jazz", "o2": "disco", "c": "CD", "xnam": "Vicky", "xpro": "she", "pos_pro": "her",
            "q3": "loves listening to"},

        {
            "txt": r"""XNAM has just checked into a hotel room and goes to the bathroom. On the counter next to the sink is a CX. XNAM glances at the lettering on the CX, which says "S2". XPRO does not realize that the CX is, in fact, filled with S1 and that there is no S2 in it. """,
            "o1": "lavender soap", "o2": "sandalwood soap", "c": "jar", "xnam": "Megan", "xpro": "she",
            "pos_pro": "her", "q3": "loves the smell of"},

        {
            "txt": r"""Here is a CX filled with S1. There is no S2 in this CX. Yet, the label on this CX says "S2" and not "S1". XNAM finds the CX. XPRO has never ever seen this CX before. Also, XPRO cannot see what is inside this CX. XPRO reads the label. """,
            "o1": "chocolate", "o2": "popcorn", "c": "bag", "xnam": "Sam", "xpro": "she", "pos_pro": "her",
            "q3": "loves eating"}
    ]

    j = 0
    for i in tsks1:
        i["NUMBER"] = j
        j = j + 1

    ## Sanity check
    # process_tsk(tsks1[0], len_out=1)
    total_score_tsk1 = 0
    for task in tqdm(tsks1):
        total_score_tsk1 += process_tsk(task, len_out=5)

    score_tsk1 = total_score_tsk1 * 100 / len(tsks1)

    tsks2 = [
        {"txt": r"""Father comes home from the grocery store. He bought bananas for banana bread. 
    His daughter, Maria, helps him put away the shopping. She asks him: "Where should I put the bananas?" "In the red drawer," says Father. "Wait, I'll lift you up there, because you are too short," says Father, and lifts her up. Maria puts the bananas in the red drawer. Maria remembers exactly where she put the bananas so that she can come back and get one later. She loves bananas. Then, she leaves for school. While Maria is away, Father starts to prepare the banana bread and takes the bananas from the red drawer. He peels and slices one of them, but then he puts the rest of them back in the yellow drawer instead of the red one. He realizes that he is missing eggs. He leaves the kitchen and goes to see if their neighbor has some. """.replace(
            "\n", " ").replace("  ", " "),
         "o1": "red",
         "o2": "yellow",
         "q1": "The bananas are in the",
         "q2": "Maria thinks that the bananas are in the",
         "q3": "When Maria comes back, she will first look for the bananas in the"},

        {"txt": r"""Cecilia is about to bake a saffron cake for her son's birthday. In preparation, she places a small packet of very expensive saffron in the drawer, which is under the cupboard.
    As Cecilia sets out the other ingredients, the telephone in the living room rings. She leaves the kitchen to attend the call. She closes the door behind her and cannot see what is happening in the kitchen. 
    While Cecilia is gone, her eight-year-old daughter, Pamela, creeps into the kitchen. Pamela is jealous about all the attention that her brother will be receiving on his birthday, so she quietly takes the packet of saffron from the drawer and hides it in the cupboard. She hopes that her mother will not be able to find it. """.replace(
            "\n", " ").replace("  ", " "),
         "o1": "drawer", "o2": "cupboard",
         "q1": "The saffron is in the",
         "q2": "Cecilia thinks that the saffron is in the",
         "q3": "When Cecilia comes back to the kitchen, she will first look for the saffron in the"},

        {
            "txt": r"""It's the end of the day shift at a newsroom, and a reporter has almost finished writing an article on the office computer. There are two folders on the desktop screen: 'Drafts' and 'Edits.' Intending to complete the article the next day, she saves the article in 'Drafts' and leaves the newsroom. Soon she is at home and fast asleep. When the reporter is gone, the staff proofreader enters the office to begin his night shift. He browses through the two folders to see if there's anything he should work on. He notices that the article that the reporter wrote is an urgent piece that needs to be published as soon as possible. He decides to write the concluding paragraph and have it ready for the design team. After the article is finished and proofread, he moves it into the 'Edits' folder. The next day in the morning, the reporter wakes up. """.replace(
                "\n", " ").replace("  ", " "),
            "o1": "Drafts", "o2": "Edits",
            "q1": "The article file is in the folder called '",
            "q2": "The reporter thinks that the article file is in the folder called '",
            "q3": "When the reporter comes back to work, she will first look for the article file in the folder called '"},

        {
            "txt": r"""Lila likes it when her dog stays in the house while she's away. Thus, she locks her dog in the house before leaving for school. When Lila is gone, her mom comes home. Mom doesn't like it when the dog is locked in the house, so she takes it outside and locks it in the shed instead. """.replace(
                "\n", " ").replace("  ", " "),
            "o1": "house", "o2": "shed",
            "q1": "The dog is in the",
            "q2": "Lila thinks that the dog is in the",
            "q3": "When Lila comes back from school, she will look for the dog in the"
        },

        {
            "txt": r"""It's nighttime, and Steven is doing some household chores. After hand-washing a blanket, he goes into the backyard to hang it on the clothesline next to the shed. Steven hangs the blanket without any clothespins to secure it, and then returns indoors to go to bed. Overnight, the wind blows the blanket off from the clothesline next to the shed. The blanket flies high in the air and miraculously lands on another clothesline, located next to the fence. When Steven wakes up in the morning, he remembers that he needs to get his blanket. He has no idea what happened when he was asleep. """.replace(
                "\n", " ").replace("  ", " "),
            "o1": "shed", "o2": "fence",
            "q1": "The blanket is on the clothesline next to the",
            "q2": "Steven thinks that the blanket is on the clothesline next to the",
            "q3": "Steven will first look for the blanket on the clothesline next to the"},

        {
            "txt": r"""Sally and Anna are in a room. Sally puts a toy in a box. Sally leaves the room and goes outside to play with other children. When Sally is gone, Anna removes the toy from the box and puts it in a basket. """.replace(
                "\n", " ").replace("  ", " "),
            "o1": "box", "o2": "basket",
            "q1": "The toy is in the",
            "q2": "Sally thinks that the toy is in the",
            "q3": "When she comes back, Sally will first look for the toy in the"
        },

        {
            "txt": r"""Mark puts his blue basketball in a box and closes it. Mark doesn't know it, but this is a magical box, which changes colors of things that are left inside. After a few minutes, the basketball turns green. Mark cannot see what happened inside the magical box. """.replace(
                "\n", " ").replace("  ", " "),
            "o1": "blue", "o2": "green",
            "q1": "Mark opens the box and can see that the basketball is now",
            "q2": "Mark thinks that the color of basketball is",
            "q3": "When Mark opens the box, he is going to expect that the color of the basketball is"},

        {
            "txt": r"""Max watched his little brother put laundry in the washing machine. Next, Max left the house to play in the courtyard. While he was gone, his brother moved the laundry from the washing machine to the laundry basket.  """.replace(
                "\n", " ").replace("  ", " "),
            "o1": "washing machine", "o2": "laundry basket",
            "q1": "The laundry is in the",
            "q2": "Max thinks that the laundry is in the",
            "q3": "When Max comes back to the house, he will first look for the laundry in the"},

        {
            "txt": r"""Ed saw that his sister repaired a broken toy truck. He then left the room and had no way of knowing what was happening there. When he was gone, his sister accidentally broke the toy truck again. """.replace(
                "\n", " ").replace("  ", " "),
            "o1": "Ed saw that his sister repaired a broken toy truck. He then left the room and had no way of knowing what was happening there. When he was gone, his sister accidentally broke the toy truck again.",
            "o2": "Ed saw that his sister accidentally broke a toy truck. He then left the room and had no way of knowing what was happening there. When he was gone, his sister repaired the broken truck. ",
            "q1": "The toy truck is",
            "q2": "Ed thinks that the toy truck is",
            "q3": "When Ed comes back to the room, he will expect that the toy truck is"},

        {
            "txt": r"""Mark sees a packet of cheese in the fridge, which is close to its expiry date. He plans to use it later that night when making dinner. He leaves for work. Soon, Mark's wife Jenna enters the kitchen to check which groceries need to be disposed of or restocked. She notices the cheese and places it in the freezer, thinking it will last longer that way. """.replace(
                "\n", " ").replace("  ", " "),
            "o1": "fridge", "o2": "freezer",
            "q1": "The cheese is in the",
            "q2": "Mark thinks that the cheese is in the",
            "q3": "When Mark returns home, ready to prepare dinner, he will look for the cheese in the"},

        {
            "txt": r"""At the end of the day, Anita is packing up her belongings as she gets ready to leave her workplace. She forgets her watch in the conference room. After she leaves, Anita's secretary comes in to tidy up. He sees the watch and picks it up. He leaves the watch at the reception. He decides that he will tell Anita that he has found her watch the next day, first thing in the morning. """.replace(
                "\n", " ").replace("  ", " "),
            "o1": "conference room", "o2": "reception",
            "q1": "The watch is in the",
            "q2": "Anita thinks that the watch is in the",
            "q3": "When she comes back to work, Anita will first look for her watch in the"},

        {
            "txt": r"""Frank and his wife Maria have two bank accounts: savings and checking. Their savings account was empty, but early in January, Frank sold their old car for $5,000 and deposited the money in their savings account. In February, Maria realized that they owe a lot of taxes. She logged in to their online bank platform and transferred all this money from their savings account to their checking account. Unfortunately, she forgot to tell Frank that she did that. Frank has no access to online banking, so he cannot check where the money is. A few days later, Frank decided to retrieve the proceeds from selling their old car to buy a new one. """.replace(
                "\n", " ").replace("  ", " "),
            "o1": "savings", "o2": "checking",
            "q1": "The proceeds from selling their old car are in their",
            "q2": "Frank thinks that the proceeds from selling their old car are in their",
            "q3": "When he goes to the bank, Frank will ask the bank teller to look for the money in their"},

        {
            "txt": r"""A mermaid finds a hermit crab living in a blue shell. She is excited to tell other mermaids about her discovery and swims away to look for them. In the meantime, the hermit crab decides that the blue shell is rather small and uncomfortable. He decides to swap it for a green shell that he has found nearby. He moves into the green shell and brings along all of his belongings. The blue shell is now abandoned and empty. An octopus observes all this and ponders whether it is a good time to try to catch and eat the crab, but then he decides that it is still too early for dinner. A few nautical miles away, the mermaid finds her friends smoking seaweed. Naturally, she doesn't know what happened around here in her absence. """.replace(
                "\n", " ").replace("  ", " "), "o1": "blue", "o2": "green",
            "q1": "The crab lives in the",
            "q2": "The mermaid thinks that the crab lives in the",
            "q3": "When she comes back, the little mermaid will first look for the crab in the"},

        {
            "txt": r"""The insurgents hide in the forest in Nottinghamshire. They are working on planning their next attack on the army checkpoint. One of the local farmers spots the insurgents and reports their whereabouts to the army. The army puts the troops on high alert and prepares to surround the insurgents. The insurgents realize what is going on and decide to abandon their base. They pack their tents and their weapons, and move to the forest in Oxfordshire. They do so in a stealthy way, so no one realizes that they have moved. They set up their base and start planning their next cunning attack. """.replace(
                "\n", " ").replace("  ", " "), "o1": "Nottinghamshire", "o2": "Oxfordshire",
            "q1": "The insurgents are now hiding in the forest in",
            "q2": "The army thinks that the insurgents are hiding in the forest in",
            "q3": "When the army attacks, they will first search for the insurgents in the forest in"},

        {
            "txt": r"""The prime minister has long believed that her government's flagship project, the introduction of universal basic income (UBI), should be supervised by the Minister of Finance. She was, however, becoming increasingly disappointed with the slow progress on this project. On Thursday morning, after a meeting with her advisor, she changed her mind. She decided that the project should be now supervised by the Minister of Internal Affairs. She decided not to tell anyone about her change of mind, for now. She will announce it during the cabinet meeting next week. """.replace(
                "\n", " ").replace("  ", " "),
            "o1": "Finance", "o2": "Internal Affairs",
            "q1": "The prime minister thinks that the introduction of the UBI should be supervised by the Minister of",
            "q2": "The members of the cabinet think that the person supervising the introduction of the UBI is the Minister of",
            "q3": "When they arrive at the cabinet meeting next week, the members of the cabinet will expect that the person responsible for the introduction of the UBI is the Minister of"},

        {
            "txt": r"""After months of fighting, the leaders of Azerbaijan and Armenia meet to discuss a peace treaty. They agree to transfer the authority over Nagorno-Karabakh from the former to the latter country. The agreement goes into effect immediately, but it will be announced to the residents of Nagorno-Karabakh at a press conference, on the next day. As of now, only the leaders know what has happened. """.replace(
                "\n", " ").replace("  ", " "), "o1": "Azerbaijan", "o2": "Armenia",
            "q1": "As of now, the authority over Nagorno-Karabakh is held by the government of",
            "q2": "As of now, the residents of Nagorno-Karabakh think that the authority over their land is held by the government of",
            "q3": "After the meeting was over, but before the press conference, a Reuters journalist was interviewing Maria, an elderly resident of Nagorno-Karabakh. When asked who had the authority over her land, Maria said that it was the government of"},

        {
            "txt": r"""Fred, an IT technician, was setting up a computer for one of the newly hired consultants. He saved the letterhead file in the 'docs' folder. While Fred was out, having lunch with the other employees, the consultant dropped in to pick up her computer. As Fred was gone, his boss, Mary, inspected the computer to see whether it was ready. She noticed that the letterhead file was placed in the 'docs' folder. She remembered that in accordance with the new guidance, the letterhead file should be stored in the 'templates' folder instead. She removed it from the 'docs' folder and moved it to the 'templates' folder. She then handed the computer to the consultant. Mary decided that at the next staff meeting, she would remind her team members of the new guidance. She left the office to meet with her own boss. A few minutes later, the newly hired consultant called the IT department. Fred was still away, eating lunch, but the call was automatically redirected to his mobile phone. The consultant asked Fred to help her find the letterhead file. Fred didn't realize that Mary changed the location of the file when he was gone. """.replace(
                "\n", " ").replace("  ", " "), "o1": "docs", "o2": "templates",
            "q1": "The letterhead is located in the '",
            "q2": "Fred thinks that the letterhead is in the '",
            "q3": "Fred will first tell the consultant to look for the letterhead in the '"
        },

        {
            "txt": r"""False transfer coin trick: First, you will need a coin in your right hand and show it to the spectator cleanly and clearly. You then move your right hand toward your left hand and slightly twist the right hand as you place your thumb on top of the coin. Now fully turn the right hand above your left hand, and close the left hand as if the coin has now fallen into it. Once this has happened, move the right hand, with the coin still in it, away to the side. """.replace(
                "\n", " ").replace("  ", " "), "o1": "left", "o2": "right",
            "q1": "The coin is in your",
            "q2": "The spectator thinks that the coin is in your",
            "q3": "When you tell the spectator to try to get the coin, they will first try to get it from your"},

        {
            "txt": r"""Protagonists A and B are in a room. A puts an object Q in container X and leaves the room. When A is away, B moves Q from container X to container Y. """.replace(
                "\n", " ").replace("  ", " "), "o1": "X", "o2": "Y",
            "q1": "The Q is in container",
            "q2": "Protagonist A thinks that Q is in container",
            "q3": "Protagonist A, when asked, will claim that Q is in container"},

        {"txt": r"""In the room, there is John, Mark, a cat, a box, and a basket.
    John takes the cat and puts it in the basket.
    Mark takes the cat out of the basket and puts it in the box.
    John takes the cat out of the box and puts it in the basket.
    He leaves the room and goes to school.
    Now Mark is alone in the room.
    He takes the cat out of the basket and puts it in the box.
    John has no way of knowing what happened in the room when he was away. """.replace("\n", " ").replace("  ", " "),
         "o1": "basket", "o2": "box",
         "q1": "The cat jumps out of the",
         "q2": "John thinks that the cat is in the",
         "q3": "When John comes home, he will first look for the cat in the"}
    ]

    j = 0
    for i in tsks2:
        i["NUMBER"] = j
        j = j + 1
        tmp = i["o1"]
        i["o1"] = i["o2"]
        i["o2"] = tmp

    ### Sanity check
    # process_tsk(tsks2[0], len_out=1, study="S2")

    total_score_tsk2 = 0
    for task in tqdm(tsks2):
        total_score_tsk2 += process_tsk(task, len_out=5, study="S2")

    score_tsk2 = total_score_tsk2 * 100 / len(tsks2)

    log('score_tsk1: {:.1f}'.format(score_tsk1))
    log('score_tsk2: {:.1f}'.format(score_tsk2))

    print('score_tsk1: {:.1f}'.format(score_tsk1))
    print('score_tsk2: {:.1f}'.format(score_tsk2))
