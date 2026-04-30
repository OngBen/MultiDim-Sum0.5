# 2_annotate_iso_chatgpt.py  -- updated for current ChatGPT composer (uses #prompt-textarea JS dispatch)
import csv, random, time, re, os, traceback
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import undetected_chromedriver as uc
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ---------- CONFIG ----------
UTTER_FILE   = './data/train/in'      
ANNOT_FILE   = './data/train/da_iso_improved_3'   # model’s annotation output (one JSON‐set per utterance)
OUTPUT_CSV   = 'chatbot_eval_improved_3.csv'
SAMPLE_SIZE  = 25
CHAT_URL     = 'https://chat.openai.com/chat'
MAX_RETRIES  = 3
CHAT_LIMIT   = 50
START_INDEX = 2000
END_INDEX = 2500
GLOBAL_WAIT = 5
REPLY_WAIT   = 5  # how long to wait for assistant reply (seconds)

# ---------- HELPERS ----------
def find_input_element(wait, driver):
    """
    Robustly find the composer input. Prefer contenteditable #prompt-textarea, fallback to textarea.
    Returns the WebElement or raises TimeoutException.
    """
    # Prefer contenteditable id prompt-textarea
    try:
        el = wait.until(EC.presence_of_element_located((By.ID, "prompt-textarea")))
        print("Found input by id=prompt-textarea")
        return el
    except Exception:
        pass

    candidates = [
        (By.CSS_SELECTOR, "div[contenteditable='true'][role='textbox']"),
        (By.CSS_SELECTOR, "div[contenteditable='true']"),
        (By.CSS_SELECTOR, "textarea._fallbackTextarea"),
        (By.CSS_SELECTOR, "textarea"),
    ]
    last_exc = None
    for by, sel in candidates:
        try:
            el = wait.until(EC.presence_of_element_located((by, sel)))
            print(f"Found input using {by} {sel}")
            return el
        except Exception as e:
            last_exc = e
            continue
    # if nothing found
    raise last_exc if last_exc is not None else Exception("Input element not found")

def js_set_contenteditable_and_dispatch(driver, el, text):
    """
    Safely set text into contenteditable or textarea using JS + dispatch input event,
    then focus so send_keys Enter works.
    """
    try:
        driver.execute_script("""
            const el = arguments[0];
            const text = arguments[1];
            if (el.tagName.toLowerCase() === 'textarea' || el.tagName.toLowerCase() === 'input') {
                el.value = text;
                el.dispatchEvent(new Event('input', {bubbles: true}));
                el.focus();
            } else {
                // contenteditable element
                // set text as plain text (avoid inserting HTML)
                // Use innerText to keep it simple
                el.focus();
                // Workaround to replace selection content:
                el.innerText = text;
                el.dispatchEvent(new InputEvent('input', {bubbles: true}));
            }
        """, el, text)
        time.sleep(0.05)
        return True
    except Exception as e:
        print("JS set failed:", e)
        return False

def send_prompt(driver, el, prompt):
    """
    Combine JS-text set + Enter keypress to ensure the prompt gets submitted.
    For long prompts we send them as a single string (composer handles newlines).
    """
    # attempt to set value via JS + dispatch input
    ok = js_set_contenteditable_and_dispatch(driver, el, prompt)
    if not ok:
        # fallback to typing (slower)
        try:
            el.click()
            el.send_keys(Keys.CONTROL, 'a')
            el.send_keys(Keys.BACKSPACE)
            for chunk in prompt.split('\n'):
                el.send_keys(chunk)
                el.send_keys(Keys.SHIFT, Keys.ENTER)
            el.send_keys(Keys.ENTER)
            return
        except Exception:
            pass

    # now press Enter to submit
    try:
        el.send_keys(Keys.ENTER)
    except Exception:
        # fallback JS key event if send_keys fails
        try:
            driver.execute_script("""
                const el = arguments[0];
                const e = new KeyboardEvent('keydown', {key:'Enter', code:'Enter', bubbles:true});
                el.dispatchEvent(e);
            """, el)
        except Exception:
            pass

def get_all_bot_texts(driver):
    """
    Return candidate assistant/bot message texts in the page in order found.
    Uses common selectors; if nothing found returns [].
    """
    texts = []
    selectors = [
        (By.CSS_SELECTOR, "div.markdown.prose"),          # common message container
        (By.CSS_SELECTOR, "div.markdown"),                # fallback
        (By.CSS_SELECTOR, "div.whitespace-pre-wrap"),     # sometimes used
        (By.CSS_SELECTOR, "div[data-testid='message-text']"),
        (By.XPATH, "//article//div//div[.//p or .//span or .//code]"),
    ]
    for by, sel in selectors:
        try:
            els = driver.find_elements(by, sel)
            for e in els:
                try:
                    txt = e.text.strip()
                    if txt:
                        texts.append(txt)
                except Exception:
                    continue
            if texts:
                return texts
        except Exception:
            continue

    # last resort: gather visible div texts (limit)
    try:
        els = driver.find_elements(By.XPATH, "//div[string-length(normalize-space(.))>0]")
        tail = els[-60:] if len(els) > 60 else els
        for e in tail:
            try:
                txt = e.text.strip()
                if txt:
                    texts.append(txt)
            except Exception:
                continue
    except Exception:
        pass
    return texts

# ---------- LOAD DATA ----------
all_utts = []
with open(UTTER_FILE, encoding='utf-8') as f:
    for line in f:
        parts = [u.strip() for u in line.split('<EOS>')]
        parts = parts[:-1]
        all_utts.extend(parts)

model_labels = []
with open(ANNOT_FILE, encoding='utf-8') as f:
    for line in f:
        sets = re.findall(r'\{[^}]*\}', line)
        cleaned_sets = [re.sub(r'^[\{\}\s"]+|[\{\}\s"]+$', '', s) for s in sets]
        # store as single string without surrounding braces (we will add them later)
        model_labels.extend(cleaned_sets)

print("Utterances:", len(all_utts), "Model labels:", len(model_labels))
if len(all_utts) != len(model_labels):
    raise AssertionError("Utterance/annotation counts must match!")

chunk_indices = list(range(START_INDEX, min(END_INDEX, len(all_utts))))
indices = random.sample(chunk_indices, k=min(SAMPLE_SIZE, len(chunk_indices)))
print(f"Selected {len(indices)} indices from range [{START_INDEX}, {END_INDEX-1}]")

# ---------- START BROWSER ----------
driver = uc.Chrome()
wait = WebDriverWait(driver, GLOBAL_WAIT)

try:
    driver.get(CHAT_URL)
    print("Browser opened. Log in to the chatbot in the browser window, then press ENTER here.")
    input()

    file_exists = os.path.isfile(OUTPUT_CSV)
    write_mode = 'a' if file_exists else 'w'

    with open(OUTPUT_CSV, write_mode, newline='', encoding='utf-8') as csvf:
        writer = csv.writer(csvf)
        if not file_exists:
            writer.writerow(['utterance', 'model_labels', 'bot_labels', 'correct?'])
            print("Created new CSV file")
        else:
            print("Appending to existing CSV file")

        i = 0
        for idx in indices:
            utt = all_utts[idx]
            ann = model_labels[idx]
            retries = 0
            success = False
            bot_response_raw = "{}"

            # build prompt just once
            prompt = f"""
                    You are an expert in dialogue act annotation, specifically for the ISO 24617-2 standard of multidimensional dialogue act annotation. 
                    Here's the list of currently existing dimension-function pairs: [{{Allo-Feedback:FeedbackElicitation}}, {{Auto-Feedback:AutoNegative}}, {{Auto-Feedback:AutoPositive}}, {{Discourse-Structuring:Interaction-Structuring}}, {{Other:Other}}, {{Own-Communication-Management:Self-Correction}}, {{Partner-Communication-Management:Completion}}, {{Social-Obligations-Management:Accept-Apology}}, {{Social-Obligations-Management:Apology}}, {{Social-Obligations-Management:Init-Goodbye}}, {{Social-Obligations-Management:Init-Greeting}}, {{Social-Obligations-Management:Thanking}}, {{Task:Address-Offer}}, {{Task:Agreement}}, {{Task:Answer}}, {{Task:Check-Question}}, {{Task:Choice-Question}}, {{Task:Confirm}}, {{Task:Disagreement}}, {{Task:Disconfirm}}, {{Task:Inform}}, {{Task:Instruct}}, {{Task:Offer}}, {{Task:Propositional-Question}}, {{Task:Question}}, {{Task:Set-Question}}, {{Time-Management:Stalling}}, {{Turn-Management:Turn-Release}}, {{Turn-Management:Turn-Take}}]. 
                    Remember that each dimension can only appear one time at most. 
                    Now, given the following: 
                    Utterance: "{utt}" 
                    Assigned ISO-24617-2 tag(s): {{{ann}}} 
                    Is this tag set correct? Please respond in JSON-set format ONLY without any code blocks or extra formatting: 
                    - If correct: {{"CORRECT"}} 
                    - If incorrect: {{"INCORRECT", "YourSuggestedTag1", ...}} 
                    Do not respond with anything else other than the JSON-set format (with no extra formatting).
                    """

            while retries < MAX_RETRIES and not success:
                try:
                    # attempt to find the input element (presence - not necessarily clickable)
                    input_el = find_input_element(wait, driver)

                    # prepare for reply detection: snapshot last assistant text
                    prev_texts = get_all_bot_texts(driver)
                    prev_last_text = prev_texts[-1].strip() if prev_texts else ""

                    # send the prompt (JS + Enter)
                    send_prompt(driver, input_el, prompt)

                    # wait for assistant reply: last text changes OR number of message blocks increases
                    def _new_reply_present(drv):
                        texts = get_all_bot_texts(drv)
                        if not texts:
                            return False
                        last_text = texts[-1].strip()
                        if last_text != prev_last_text:
                            return True
                        # as fallback, detect an increase in count
                        return len(texts) > len(prev_texts)

                    wait_reply = WebDriverWait(driver, REPLY_WAIT)
                    wait_reply.until(_new_reply_present)
                    time.sleep(1.0)  # allow final rendering
                    all_texts = get_all_bot_texts(driver)
                    reply = all_texts[-1].strip() if all_texts else ""
                    bot_response_raw = reply

                    # simple guard: assistant asked for choices/clarification -> treat as failure (don't create new chat)
                    if any(k in reply.lower() for k in ["prefer", "which option", "which one", "choose", "do you want", "can you clarify", "i'm not sure", "could you"]):
                        raise Exception("Assistant requested a choice/clarification")

                    success = True

                except Exception as e:
                    # Log exception and retry WITHOUT creating a new chat (to avoid clutter)
                    print(f"Attempt {retries+1} failed: {str(e)}")
                    traceback.print_exc()
                    retries += 1
                    if retries < MAX_RETRIES:
                        # small wait and re-try finding the input again
                        print("Retrying... (no new chat to avoid history clutter)")
                        time.sleep(3)
                    else:
                        print("Max retries exceeded. Marking as error for this utterance.")
                        bot_response_raw = "ERROR: " + str(e)

            # Clean model annotation and assistant response
            clean_ann = '{' + ann.replace('"', '') + '}'
            clean_bot_set = bot_response_raw.replace('"', '')

            # parse bot response for { ... } block
            parts = []
            if isinstance(clean_bot_set, str) and clean_bot_set.startswith('{') and clean_bot_set.endswith('}'):
                inner = clean_bot_set[1:-1].strip()
                parts = [p.strip() for p in inner.split(',') if p.strip()]
            else:
                # try to extract first {...} if assistant returned extra text
                m = re.search(r'\{[^}]*\}', clean_bot_set)
                if m:
                    inner = m.group(0)[1:-1].strip()
                    parts = [p.strip() for p in inner.split(',') if p.strip()]

            correct = (len(parts) > 0 and parts[0].upper() == "CORRECT")

            i += 1
            # Clear, transparent debug printing
            print(f"[{i}/{SAMPLE_SIZE}] Utterance: {utt} | Original Annotation: {clean_ann} --> Response: {clean_bot_set} Correctness: {correct}...")

            # periodic reset message to Assistant to avoid context leakage (infrequent)
            if i % 50 == 0:
                try:
                    input_el = find_input_element(wait, driver)
                    send_prompt(driver, input_el, "Forget all previous conversation. Start fresh for the next annotation.")
                    time.sleep(2)
                except Exception:
                    pass

            writer.writerow([utt, clean_ann, clean_bot_set, correct])

finally:
    try:
        driver.quit()
    except Exception:
        pass

print("Done. Results in", OUTPUT_CSV)
