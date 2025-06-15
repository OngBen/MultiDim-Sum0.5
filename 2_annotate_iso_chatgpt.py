import csv, random, time, re, os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import undetected_chromedriver as uc
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
# ---------- CONFIG ----------
UTTER_FILE   = './data/valid/in'      # your original utterance file
ANNOT_FILE   = './data/valid/da_iso'   # model’s annotation output (one JSON‐set per utterance)
OUTPUT_CSV   = 'chatbot_eval.csv'
SAMPLE_SIZE  = 50
CHAT_URL     = 'https://chatgpt.com'  # or other free chatbot URL
MAX_RETRIES  = 3  # Max retries per utterance
CHAT_LIMIT   = 50
START_INDEX = 1500
END_INDEX = 2000

# ---------- LOAD DATA ----------
# 1) extract all utterances (split on <EOS>)
all_utts = []
with open(UTTER_FILE, encoding='utf-8') as f:
    for line in f:
        parts = [u.strip() for u in line.split('<EOS>')]
        parts = parts[:-1]
        all_utts.extend(parts)

# 2) load model annotations (one line = same number of JSON‐sets as utterances on that line)
model_labels = []
with open(ANNOT_FILE, encoding='utf-8') as f:
    for line in f:
        # extract each {...} block
        sets = re.findall(r'\{[^}]*\}', line)
        # Remove all quotes and braces from each set
        cleaned_sets = [re.sub(r'[\"{}]', '', s) for s in sets]
        model_labels.extend(cleaned_sets)
print(len(all_utts))
print(len(model_labels))
assert len(all_utts) == len(model_labels), \
    "Utterance/annotation counts must match!"

# 3) sample indices
chunk_indices = list(range(START_INDEX, END_INDEX))
indices = random.sample(chunk_indices, k=min(SAMPLE_SIZE, len(chunk_indices)))
print(f"Selected {len(indices)} indices from range [{START_INDEX}, {END_INDEX-1}]")

# ---------- START BROWSER ----------
driver = uc.Chrome()
driver.get(CHAT_URL)
print("Log in to the chatbot, then press ENTER in this console.")
input()

file_exists = os.path.isfile(OUTPUT_CSV)
write_mode = 'a' if file_exists else 'w'

#{{"Task":"Inform"}},{{"Task":"Agreement"}},{{"Task":"Disagreement"}},{{"Task":"Correction"}},{{"Task":"Answer"}},{{"Task":"Confirm"}},{{"Task":"Disconfirm"}},{{"Task":"Question"}},{{"Task":"Set-Question"}},{{"Task":"Propositional-Question"}},{{"Task":"Choice-Question"}},{{"Task":"Check-Question"}},{{"Task":"Offer"}},{{"Task":"Address-Offer"}},{{"Task":"Accept-Offer"}},{{"Task":"Decline-Offer"}},{{"Task":"Promise"}},{{"Task":"Request"}},{{"Task":"Address-Request"}},{{"Task":"Accept-Request"}},{{"Task":"Decline-Request"}},{{"Task":"Suggest"}},{{"Task":"Address-Suggest"}},{{"Task":"Accept-Suggest"}},{{"Task":"Decline-Suggest"}},{{"Task":"Instruct"}},{{"Auto-Feedback":"AutoPositive"}},{{"Auto-Feedback":"AutoNegative"}},{{"Allo-Feedback":"AlloPositive"}},{{"Allo-Feedback":"AlloNegative"}},{{"Allo-Feedback":"FeedbackElicitation"}},{{"Time-Management":"Stalling"}},{{"Time-Management":"Pausing"}},{{"Turn-Management":"Turn-Take"}},{{"Turn-Management":"Turn-Grab"}},{{"Turn-Management":"Turn-Accept"}},{{"Turn-Management":"Turn-Keep"}},{{"Turn-Management":"Turn-Give"}},{{"Turn-Management":"Turn-Release"}},{{"Own-Communication-Management":"Self-Correction"}},{{"Own-Communication-Management":"Self-Error"}},{{"Own-Communication-Management":"Retraction"}},{{"Partner-Communication-Management":"Completion"}},{{"Partner-Communication-Management":"Correct-Misspeaking"}},{{"Social-Obligations-Management":"Init-Greeting"}},{{"Social-Obligations-Management":"Return-Greeting"}},{{"Social-Obligations-Management":"Init-Self-Introduction"}},{{"Social-Obligations-Management":"Return-Self-Introduction"}},{{"Social-Obligations-Management":"Apology"}},{{"Social-Obligations-Management":"Accept-Apology"}},{{"Social-Obligations-Management":"Thanking"}},{{"Social-Obligations-Management":"Accept-Thanking"}},{{"Social-Obligations-Management":"Goodbye"}},{{"Social-Obligations-Management":"Return-Goodbye"}},{{"Discourse-Structuring":"Interaction-Structuring"}},{{"Other":"Other"}}

# ---------- EVALUATION LOOP ----------
with open(OUTPUT_CSV, write_mode, newline='', encoding='utf-8') as csvf:
    writer = csv.writer(csvf)
    if not file_exists:
        writer.writerow(['utterance', 'model_labels', 'bot_labels', 'correct?'])
        print("Created new CSV file")
    else:
        print("Appending to existing CSV file")

    wait = WebDriverWait(driver, 30)
    i = 0
    for idx in indices:
        utt = all_utts[idx]
        ann = model_labels[idx]  # e.g. '"Task:Inform","Auto-Feedback:AutoPositive"'
        retries = 0
        success = False
        bot_response = ""

        while retries < MAX_RETRIES and not success:
            try:
                if i % CHAT_LIMIT == 0:
                    wait.until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, '[data-testid="create-new-chat-button"]'))
                    ).click()

                prompt = f"""
                You are an expert in dialogue act annotation, specifically for the ISO 24617-2 standard of multidimensional dialogue act annotation.
                Here's the list of currently existing dimension-function pairs: [{{Allo-Feedback:FeedbackElicitation}}, {{Auto-Feedback:AutoNegative}}, {{Auto-Feedback:AutoPositive}}, {{Discourse-Structuring:Interaction-Structuring}}, {{Other:Other}}, {{Own-Communication-Management:Self-Correction}}, {{Partner-Communication-Management:Completion}}, {{Social-Obligations-Management:Accept-Apology}}, {{Social-Obligations-Management:Apology}}, {{Social-Obligations-Management:Init-Goodbye}}, {{Social-Obligations-Management:Init-Greeting}}, {{Social-Obligations-Management:Thanking}}, {{Task:Address-Offer}}, {{Task:Agreement}}, {{Task:Answer}}, {{Task:Check-Question}}, {{Task:Choice-Question}}, {{Task:Confirm}}, {{Task:Disagreement}}, {{Task:Disconfirm}}, {{Task:Inform}}, {{Task:Instruct}}, {{Task:Offer}}, {{Task:Propositional-Question}}, {{Task:Question}}, {{Task:Set-Question}}, {{Time-Management:Stalling}}, {{Turn-Management:Turn-Release}}, {{Turn-Management:Turn-Take}}].
                Remember that each dimension can only appear one time at most.
                Now, given the following:
                Utterance: "{utt}"
                Assigned ISO-24617-2 tag(s): {{{ann}}}
                Is this tag set correct? Please respond in JSON-set format ONLY:
                - If correct: {{"CORRECT"}}
                - If incorrect: {{"INCORRECT", "YourSuggestedTag1", ...}}
                Do not respond with anything else other than the JSON-set format.
                """
                print("1")        
                # Wait for the prompt box to appear and become clickable
                input_div = wait.until(
                    EC.element_to_be_clickable((By.ID, "prompt-textarea"))
                )
                print("2")
                input_div.click()
                input_div.clear()

                lines = prompt.split("\n")


                # send each line with SHIFT+ENTER between them
                for line_idx, line in enumerate(lines):
                    input_div.send_keys(line)
                    if line_idx < len(lines) - 1:
                        input_div.send_keys(Keys.SHIFT, Keys.ENTER)
                input_div.send_keys(Keys.ENTER)

                print("3")
                # wait for response
                time.sleep(10)  
                print("4")
                # grab latest response
                replies = driver.find_elements(By.CSS_SELECTOR, ".markdown.prose")
                reply = replies[-1].text.strip() if replies else ""

                if "prefer" in reply.lower() or "option" in reply.lower():
                    raise Exception("ChatGPT requested choice between responses")

                # Parse chatbot response
                match = re.search(r'\{.*\}', reply)
                bot_set_raw = match.group(0) if match else "{}"

                bot_response = match.group(0)
                success = True

            except Exception as e:
                print(f"Attempt {retries+1} failed: {str(e)}")
                retries += 1
                if retries < MAX_RETRIES:
                    print("Retrying...")
                    # Reset chat explicitly before retry
                    wait.until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, '[data-testid="create-new-chat-button"]'))
                    ).click()
                    time.sleep(3)
                else:
                    print("Max retries exceeded. Marking as error.")
                    bot_response = "ERROR: " + str(e)
                
        # Clean model annotation (remove quotes, add braces)
        clean_ann = '{' + ann.replace('"', '') + '}'
        # Clean bot response (remove quotes but keep braces)
        clean_bot_set = bot_set_raw.replace('"', '')

        # Determine correctness (unchanged)
        if clean_bot_set.startswith('{') and clean_bot_set.endswith('}'):
            inner = clean_bot_set[1:-1].strip()
            parts = [p.strip() for p in inner.split(',') if p.strip()]
        else:
            parts = []
        
        # Check correctness: first element must be "CORRECT"
        correct = (parts and parts[0] == "CORRECT")

        i+=1
        print(f"[{i}/{SAMPLE_SIZE}] Utterance: {utt} | Original Annotation: {clean_ann} --> Response: {clean_bot_set} Correctness: {correct}...")

        if i % 20 == 0:
            input_div = wait.until(EC.element_to_be_clickable((By.ID, "prompt-textarea")))
            input_div.click()
            input_div.send_keys("Forget all previous conversation. Start fresh for the next annotation.")
            input_div.send_keys(Keys.ENTER)
            time.sleep(5)

        writer.writerow([utt, clean_ann, clean_bot_set, correct])
driver.quit()
print("Done. Results in", OUTPUT_CSV)
