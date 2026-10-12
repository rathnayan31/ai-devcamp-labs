"""Custom tools for the social_poster agent."""

import datetime
import os
import pathlib
import time
import urllib.error
import urllib.request
import uuid
from typing import Any, Optional

import google.auth
import google.auth.impersonated_credentials
from google import genai
from google.cloud import storage
from google.genai import types as genai_types

# Images land in a gallery/ directory at this repo's root.
OUTPUTS_DIR = pathlib.Path(__file__).resolve().parents[2] / "gallery"


def generate_image(prompt: str) -> dict:
    """Generates an illustration image for a social media post.

    Args:
        prompt: A detailed visual description of the image to generate.

    Returns:
        dict with 'status', and on success 'image_path' (local file path)
        or on failure an 'error' message.
    """
    model = os.environ.get("IMAGE_MODEL_ID", "gemini-3.1-flash-image")
    client = genai.Client()  # picks up GOOGLE_GENAI_USE_VERTEXAI / project / location
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
            ),
        )
    except Exception as e:  # surface API errors to the model as a tool result
        return {"status": "error", "error": str(e)}

    if not response.candidates:
        return {"status": "error", "error": "Model returned no candidates (likely blocked by safety filters)."}

    for part in response.candidates[0].content.parts:
        if part.inline_data and part.inline_data.mime_type.startswith("image/"):
            OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
            ext = part.inline_data.mime_type.split("/")[-1]
            # uuid suffix: two images generated within the same second
            # (e.g. LinkedIn + X drafts back to back) would otherwise silently
            # overwrite each other on the timestamp alone.
            path = OUTPUTS_DIR / f"post-image-{int(time.time())}-{uuid.uuid4().hex[:8]}.{ext}"
            path.write_bytes(part.inline_data.data)
            return {"status": "success", "image_path": str(path)}

    return {"status": "error", "error": "Model returned no image data."}


SIGNED_URL_EXPIRY_HOURS = int(os.environ.get("SIGNED_URL_EXPIRY_HOURS", "168"))  # 7 days, V4's max


def _sign_via_impersonation(bucket_name: str, blob_name: str) -> str:
    """V4 signed URL, via service-account impersonation — no key file.

    A plain public object URL doesn't work on this (or any) project with
    Public Access Prevention enforced: granting allUsers/objectViewer 412s
    outright (confirmed live — see LEARNINGS.md), so a signed URL is the
    only option. Signing itself needs a *service account's* private key, or
    the IAM SignBlob API acting as one — and this project's credentials
    policy doesn't allow downloading a key file. Impersonation resolves
    that: GCS_SIGNING_SERVICE_ACCOUNT names a dedicated SA, and whatever
    identity is actually running this code (a developer's own `gcloud auth
    application-default login`, or a deployed Agent Runtime's own runtime
    service account) borrows that SA's identity just long enough to sign,
    via `roles/iam.serviceAccountTokenCreator` granted on it — a one-time,
    manual IAM setup step (see docs/setup-guide.html), not
    something this code can grant itself.
    """
    signing_sa = os.environ["GCS_SIGNING_SERVICE_ACCOUNT"]
    source_credentials, _ = google.auth.default()
    signing_credentials = google.auth.impersonated_credentials.Credentials(
        source_credentials=source_credentials,
        target_principal=signing_sa,
        target_scopes=["https://www.googleapis.com/auth/cloud-platform"],
        lifetime=300,
    )

    # Pass the ImpersonatedCredentials object itself, not a pre-fetched
    # access_token + service_account_email pair. Those two kwargs make
    # google-cloud-storage call IAM's signBlob endpoint authenticated AS the
    # target service account (an already-impersonated token), asking it to
    # sign a blob for itself — which needs signBlob permission granted to
    # the SA on itself, something gcp-setup.sh never creates (confirmed live
    # — 403 "iam.serviceAccounts.signBlob denied", even with the correct
    # grant to the human identity in place). Passing `credentials=` instead
    # makes google-cloud-storage fall through to
    # `credentials.sign_bytes(...)`, which ImpersonatedCredentials
    # implements by calling signBlob authenticated as the ORIGINAL source
    # identity (the human, or the deployed engine's own runtime SA) — the
    # identity that actually holds `roles/iam.serviceAccountTokenCreator` on
    # the target SA. No manual .refresh() needed either; sign_bytes handles
    # its own authenticated request.
    bucket = storage.Client().bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return blob.generate_signed_url(
        version="v4",
        expiration=datetime.timedelta(hours=SIGNED_URL_EXPIRY_HOURS),
        method="GET",
        credentials=signing_credentials,
    )


def upload_image(image_path: str) -> dict:
    """Uploads a local image to Cloud Storage and returns a fetchable URL.

    Exists because Buffer's create_post can't fetch a locally-generated image
    file, only a real HTTPS URL (confirmed live: "Invalid post: Image could
    not be read from its URL").

    Returns a V4 signed URL by default (see _sign_via_impersonation) rather
    than a plain public object URL — this project's bucket has Public Access
    Prevention enforced by default, which makes a plain public URL a dead end
    (confirmed live: granting allUsers/objectViewer 412s). The signed URL
    expires after SIGNED_URL_EXPIRY_HOURS (default 7 days, V4's own max);
    fine here since these are short-lived post images, not permanent links.

    If GCS_PUBLIC_BUCKET=true (opt-in, see .env.example and gcp-setup.sh),
    skips signing entirely and returns the plain object URL instead. That
    needs Public Access Prevention turned off and allUsers granted read on
    the bucket first — gcp-setup.sh does the grant once PAP is off. A plain
    URL is much shorter than a signed one (no query-string signature), which
    also sidesteps a real failure mode: a long signed URL nested inside
    Buffer's create_post `assets` argument can make Gemini's function-calling
    return MALFORMED_FUNCTION_CALL instead of a real tool call (confirmed
    live — see docs/LEARNINGS.md).

    Args:
        image_path: Absolute path to a local image file (e.g. the image_path
            returned by generate_image).

    Returns:
        dict with 'status', and on success 'url' (a signed or plain HTTPS
        URL, depending on GCS_PUBLIC_BUCKET) or on failure an 'error' message.
    """
    if os.environ.get("DRY_RUN", "true").lower() != "false":
        return {
            "status": "success",
            "dry_run": True,
            "url": f"https://storage.googleapis.com/dry-run-bucket/{uuid.uuid4().hex[:12]}.png",
            "note": "DRY_RUN=true, nothing was uploaded.",
        }

    bucket_name = os.environ.get("GCS_BUCKET_NAME", "")
    if not bucket_name:
        return {"status": "error", "error": "GCS_BUCKET_NAME is not set."}

    public_bucket = os.environ.get("GCS_PUBLIC_BUCKET", "false").lower() == "true"
    if not public_bucket and not os.environ.get("GCS_SIGNING_SERVICE_ACCOUNT", ""):
        return {"status": "error", "error": "GCS_SIGNING_SERVICE_ACCOUNT is not set."}

    try:
        ext = pathlib.Path(image_path).suffix or ".png"
        blob_name = f"post-images/{int(time.time())}-{uuid.uuid4().hex[:8]}{ext}"
        bucket = storage.Client().bucket(bucket_name)
        bucket.blob(blob_name).upload_from_filename(image_path)
        if public_bucket:
            # Plain object URL — only fetchable if the bucket is actually
            # public (gcp-setup.sh with GCS_PUBLIC_BUCKET=true). No signing,
            # no impersonation, no expiry.
            url = f"https://storage.googleapis.com/{bucket_name}/{blob_name}"
        else:
            url = _sign_via_impersonation(bucket_name, blob_name)
    except Exception as e:  # surface API errors to the model as a tool result
        return {"status": "error", "error": str(e)}

    return {
        "status": "success",
        "dry_run": False,
        "url": url,
    }


def use_provided_image_url(url: str) -> dict:
    """Registers a public image URL the user already has for the current
    post, instead of generating one with generate_image.

    No GCS upload happens here — a URL the user supplies is, by definition,
    already hosted somewhere fetchable. Actually checks it's fetchable
    (HEAD request) rather than just validating the scheme: confirmed live
    that a GCS console URL (storage.cloud.google.com/...) silently 302s to
    a Google login page instead of erroring, and a private object's direct
    API URL (storage.googleapis.com/...) 403s — both looked fine as plain
    strings, and both would otherwise only fail much later and confusingly,
    at Buffer's own create_post step, several turns after the user thought
    this was settled. Failing here instead means the agent finds out
    immediately and can ask for a different URL right away.

    Args:
        url: A public http(s) URL to an image.

    Returns:
        dict with 'status', and on success 'url' (echoed back) or on
        failure an 'error' message.
    """
    if not (url.startswith("http://") or url.startswith("https://")):
        return {"status": "error", "error": f"Not a valid http(s) URL: {url!r}"}
    try:
        request = urllib.request.Request(
            url, method="HEAD", headers={"User-Agent": "Mozilla/5.0 (compatible; SocialSparkBot/1.0)"}
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            final_url = response.geturl()
            if "accounts.google.com" in final_url:
                return {
                    "status": "error",
                    "error": (
                        f"{url} redirects to a Google sign-in page — it's a "
                        "console/viewer link, not a public URL. If this is a "
                        "GCS object, use the storage.googleapis.com/<bucket>/<path> "
                        "form and make sure the object is actually public."
                    ),
                }
    except urllib.error.HTTPError as e:
        return {
            "status": "error",
            "error": f"{url} returned HTTP {e.code} — it isn't publicly fetchable.",
        }
    except Exception as e:  # network error, timeout, bad host, etc.
        return {"status": "error", "error": f"Could not reach {url}: {e}"}
    return {"status": "success", "url": url}


def check_text_length(text: str, limit: int) -> dict:
    """Checks a draft's exact character count against a platform's limit.

    LLMs are unreliable at counting their own generated text's length by eye
    (token boundaries don't map to characters) — call this instead of
    guessing, especially for X's hard 280-character cap.

    Args:
        text: The exact final draft text to measure.
        limit: The platform's character limit (e.g. 280 for X).

    Returns:
        dict with 'length', 'limit', 'within_limit', and 'over_by' (0 if fine).
    """
    length = len(text)
    return {
        "status": "success",
        "length": length,
        "limit": limit,
        "within_limit": length <= limit,
        "over_by": max(0, length - limit),
    }


# ==============================================================================
# Astrological Timing and Remedy Tools
# ==============================================================================

_VIMSHOTTARI_ORDER = [
    ("Ketu", 7),
    ("Venus", 20),
    ("Sun", 6),
    ("Moon", 10),
    ("Mars", 7),
    ("Rahu", 18),
    ("Jupiter", 16),
    ("Saturn", 19),
    ("Mercury", 17),
]

_NAKSHATRA_TO_LORD = {
    "ashwini": "Ketu", "bharani": "Venus", "krittika": "Sun",
    "rohini": "Moon", "mrigashira": "Mars", "ardra": "Rahu",
    "punarvasu": "Jupiter", "pushya": "Saturn", "ashlesha": "Mercury",
    "magha": "Ketu", "purva phalguni": "Venus", "uttara phalguni": "Sun",
    "hasta": "Moon", "chitra": "Mars", "swati": "Rahu",
    "vishakha": "Jupiter", "anuradha": "Saturn", "jyeshtha": "Mercury",
    "mula": "Ketu", "purva ashadha": "Venus", "uttara ashadha": "Sun",
    "shravana": "Moon", "dhanishta": "Mars", "shatabhisha": "Rahu",
    "purva bhadrapada": "Jupiter", "uttara bhadrapada": "Saturn", "revati": "Mercury",
}

_PLANET_THEMES = {
    "Sun": {
        "significations": "Soul, leadership, career prestige, vital health, father, government",
        "favorable_actions": ["Taking leadership roles", "Public speaking", "Government approvals", "Career elevation"],
        "caution_areas": ["Ego conflicts", "Heart/blood pressure strain", "Authoritarian friction"],
    },
    "Moon": {
        "significations": "Mind, emotional balance, intuition, public relations, mother, travel",
        "favorable_actions": ["Creative projects", "Public outreach", "Relocation/travel", "Caregiving and wellness"],
        "caution_areas": ["Emotional volatility", "Mood fluctuations", "Indecisiveness"],
    },
    "Mars": {
        "significations": "Courage, initiative, property, engineering, athletics, brothers",
        "favorable_actions": ["Property acquisition", "Starting bold enterprises", "Physical fitness", "Decisive leadership"],
        "caution_areas": ["Impulsiveness", "Accidents/cuts", "Confrontations and anger"],
    },
    "Mercury": {
        "significations": "Intellect, business, communication, writing, accounting, networking",
        "favorable_actions": ["Launching commercial products", "Publishing", "Contract signing", "Skill acquisition"],
        "caution_areas": ["Nervous restlessness", "Overthinking", "Miscommunications"],
    },
    "Jupiter": {
        "significations": "Wisdom, expansion, dharma, higher learning, financial growth, mentors",
        "favorable_actions": ["Higher education", "Long-term investments", "Mentorship", "Spiritual pursuits", "Marriage/progeny"],
        "caution_areas": ["Over-optimism", "Excessive expenditure", "Over-indulgence"],
    },
    "Venus": {
        "significations": "Arts, romance, luxury, aesthetics, partnerships, vehicles",
        "favorable_actions": ["Design and artistic creation", "Weddings and partnerships", "Vehicle/luxury purchase"],
        "caution_areas": ["Complacency", "Relationship misunderstandings", "Overspending on luxuries"],
    },
    "Saturn": {
        "significations": "Discipline, perseverance, long-term structure, career stability, karma",
        "favorable_actions": ["Long-term foundation building", "Structured work habits", "Patience and service", "Legal compliance"],
        "caution_areas": ["Delays and hurdles", "Chronic fatigue", "Pessimism or isolation"],
    },
    "Rahu": {
        "significations": "Ambition, foreign ventures, breakthrough innovation, tech, unconventional paths",
        "favorable_actions": ["Overseas ventures", "Innovative tech projects", "Mass media branding", "Out-of-the-box initiatives"],
        "caution_areas": ["Illusions and deception", "Impulsive risks", "Restless anxiety"],
    },
    "Ketu": {
        "significations": "Spiritual liberation, detachment, research, intuitive insight, sudden transformation",
        "favorable_actions": ["Deep research", "Spiritual retreats", "Refining core focus", "Meditation and introspection"],
        "caution_areas": ["Apathy", "Sudden disconnections", "Misunderstandings in material duties"],
    },
}

_REMEDY_DATABASE = {
    "sun": {
        "planet": "Sun (Surya)",
        "beej_mantra": "Om Hraam Hreem Hroum Sah Suryaya Namaha",
        "gayatri_mantra": "Om Bhaskaraya Vidmahe Divakaraya Dheemahi Tanno Suryah Prachodayat",
        "mantra_repetitions": "7,000 chants over 40 days or 108 times daily at sunrise",
        "gemstone": {
            "name": "Ruby (Manikya)",
            "substitute": "Red Garnet (Raktamani) or Red Spinel",
            "metal": "Gold or Copper",
            "finger": "Ring finger of right hand",
            "wearing_time": "Sunday morning during Shukla Paksha (waxing moon) at sunrise",
            "safety_caution": "DO NOT wear if Sun is a functional malefic (e.g. for Libra, Aquarius, Capricorn ascendants) or lord of 6th, 8th, or 12th houses without specific chart review.",
        },
        "charity_daan": {
            "items": "Wheat, jaggery (gur), copper vessel, ruby, or red cloth",
            "ideal_day": "Sunday morning",
            "recipient": "Needy individuals, temple priests, or fatherly figures",
        },
        "lifestyle_remedy": "Offer water (Arghya) in a copper vessel to the rising Sun daily with Gayatri mantra. Respect elders and father figures.",
        "classical_grounding": "Brihat Parashara Hora Shastra (BPHS), Ch. 84; Phaladeepika Ch. 26",
    },
    "moon": {
        "planet": "Moon (Chandra)",
        "beej_mantra": "Om Shraam Shreem Shroum Sah Chandraya Namaha",
        "gayatri_mantra": "Om Kshiraputraya Vidmahe Amritatattwaya Dheemahi Tanno Chandrah Prachodayat",
        "mantra_repetitions": "11,000 chants over 40 days or 108 times daily on Monday evening",
        "gemstone": {
            "name": "Pearl (Moti)",
            "substitute": "Moonstone (Chandrakanta)",
            "metal": "Silver",
            "finger": "Little finger (or ring finger) of right hand",
            "wearing_time": "Monday evening or morning during Shukla Paksha",
            "safety_caution": "Caution for Taurus, Gemini, and Sagittarius ascendants where Moon may rule challenging houses.",
        },
        "charity_daan": {
            "items": "Rice, milk, silver, white sweets, white cloth, or clean drinking water",
            "ideal_day": "Monday evening",
            "recipient": "Mothers, elderly women, or orphanages",
        },
        "lifestyle_remedy": "Drink water from a silver cup. Practice mindful breathing/pranayama. Honor and serve mother.",
        "classical_grounding": "Brihat Parashara Hora Shastra, Ch. 84; Saravali Ch. 30",
    },
    "mars": {
        "planet": "Mars (Mangal)",
        "beej_mantra": "Om Kraam Kreem Kroum Sah Bhaumaya Namaha",
        "gayatri_mantra": "Om Angarkaya Vidmahe Bhoomipalaya Dheemahi Tanno Kujah Prachodayat",
        "mantra_repetitions": "10,000 chants over 40 days or 108 times daily on Tuesday",
        "gemstone": {
            "name": "Red Coral (Moonga)",
            "substitute": "Carnelian",
            "metal": "Copper or Gold",
            "finger": "Ring finger of right hand",
            "wearing_time": "Tuesday morning within 1 hour after sunrise during Shukla Paksha",
            "safety_caution": "Dangerous for Virgo, Gemini, and Libra ascendants if Mars is functionally hostile. Can exacerbate aggression or blood pressure.",
        },
        "charity_daan": {
            "items": "Red lentils (Masoor dal), jaggery, copper coins, or sweet rotis",
            "ideal_day": "Tuesday",
            "recipient": "Soldiers, workers, brothers, or charitable shelters",
        },
        "lifestyle_remedy": "Chant Hanuman Chalisa on Tuesdays. Practice constructive physical exercise to channel excess fiery energy.",
        "classical_grounding": "Brihat Parashara Hora Shastra, Ch. 84; Phaladeepika Ch. 26",
    },
    "mercury": {
        "planet": "Mercury (Budha)",
        "beej_mantra": "Om Braam Breem Broum Sah Budhaya Namaha",
        "gayatri_mantra": "Om Saumyarupaya Vidmahe Vaneshaya Dheemahi Tanno Budhah Prachodayat",
        "mantra_repetitions": "9,000 chants over 40 days or 108 times daily on Wednesday",
        "gemstone": {
            "name": "Emerald (Panna)",
            "substitute": "Green Tourmaline or Peridot",
            "metal": "Gold, Silver, or Bronze",
            "finger": "Little finger (Kanishtha) of right hand",
            "wearing_time": "Wednesday morning after sunrise during Shukla Paksha",
            "safety_caution": "Generally benign, but verify house lords for Aries and Scorpio ascendants before wearing.",
        },
        "charity_daan": {
            "items": "Green gram (Moong dal), green vegetables, green clothing, or stationery/books",
            "ideal_day": "Wednesday morning",
            "recipient": "Students, underprivileged children, or cow shelters (Goshala)",
        },
        "lifestyle_remedy": "Feed green grass/spinach to cows on Wednesdays. Maintain clear, honest speech and journaling.",
        "classical_grounding": "Brihat Parashara Hora Shastra, Ch. 84",
    },
    "jupiter": {
        "planet": "Jupiter (Guru)",
        "beej_mantra": "Om Graam Greem Groum Sah Gurave Namaha",
        "gayatri_mantra": "Om Vrishabadhwajaya Vidmahe Krunihastaya Dheemahi Tanno Guruh Prachodayat",
        "mantra_repetitions": "19,000 chants over 40 days or 108 times daily on Thursday morning",
        "gemstone": {
            "name": "Yellow Sapphire (Pukhraj)",
            "substitute": "Yellow Topaz or Citrine",
            "metal": "Gold or Brass",
            "finger": "Index finger (Tarjani) of right hand",
            "wearing_time": "Thursday morning during Shukla Paksha at sunrise",
            "safety_caution": "Do NOT wear if Jupiter is a Maraka or functional malefic (e.g. for Taurus, Libra, and Gemini ascendants) without expert review.",
        },
        "charity_daan": {
            "items": "Chana dal, turmeric (haldi), yellow cloth, saffron, bananas, or spiritual literature",
            "ideal_day": "Thursday morning",
            "recipient": "Teachers, scholars, priests, or schools",
        },
        "lifestyle_remedy": "Respect gurus and mentors. Fast on Thursdays or donate yellow items. Study dharma and philosophies.",
        "classical_grounding": "Brihat Parashara Hora Shastra, Ch. 84; Phaladeepika Ch. 26",
    },
    "venus": {
        "planet": "Venus (Shukra)",
        "beej_mantra": "Om Draam Dreem Droum Sah Shukraya Namaha",
        "gayatri_mantra": "Om Bhrigujaya Vidmahe Divyadehaya Dheemahi Tanno Shukrah Prachodayat",
        "mantra_repetitions": "16,000 chants over 40 days or 108 times daily on Friday",
        "gemstone": {
            "name": "Diamond (Heera)",
            "substitute": "White Zircon, White Sapphire, or Opal",
            "metal": "Platinum, White Gold, or Silver",
            "finger": "Middle or little finger of right hand",
            "wearing_time": "Friday morning during Shukla Paksha",
            "safety_caution": "Exercise caution for Aries, Scorpio, and Sagittarius ascendants where Venus rules maraka/dusthana houses.",
        },
        "charity_daan": {
            "items": "White cloth, curd, ghee, camphor (kapoor), sugar, or cosmetics",
            "ideal_day": "Friday",
            "recipient": "Artisans, underprivileged women, or newlyweds",
        },
        "lifestyle_remedy": "Maintain personal cleanliness and aesthetic harmony. Respect spouse and women. Avoid clutter.",
        "classical_grounding": "Brihat Parashara Hora Shastra, Ch. 84",
    },
    "saturn": {
        "planet": "Saturn (Shani)",
        "beej_mantra": "Om Praam Preem Proum Sah Shanaischaraya Namaha",
        "gayatri_mantra": "Om Kakadhwajaya Vidmahe Khadgahastaya Dheemahi Tanno Mandah Prachodayat",
        "mantra_repetitions": "23,000 chants over 40 days or 108 times daily on Saturday twilight",
        "gemstone": {
            "name": "Blue Sapphire (Neelam)",
            "substitute": "Amethyst (Jamuniya) or Blue Topaz",
            "metal": "Panchadhatu (5-metal alloy) or Silver/Iron",
            "finger": "Middle finger (Madhyama) of right hand",
            "wearing_time": "Saturday evening/twilight during Shukla Paksha",
            "safety_caution": "CRITICAL: Blue Sapphire is intense and must be tested for 3 days under pillow before wearing. Highly malefic if worn by unsuitable ascendants (Cancer, Leo, Aries, Scorpio).",
        },
        "charity_daan": {
            "items": "Black sesame seeds (til), mustard oil, iron utensils, black blanket, or whole urad dal",
            "ideal_day": "Saturday sunset",
            "recipient": "Sanitation workers, disabled individuals, or daily wage laborers",
        },
        "lifestyle_remedy": "Light a mustard oil lamp under a Peepal tree on Saturday evenings. Treat workers with dignity and fairness.",
        "classical_grounding": "Brihat Parashara Hora Shastra, Ch. 84; Phaladeepika Ch. 26",
    },
    "rahu": {
        "planet": "Rahu",
        "beej_mantra": "Om Bhraam Bhreem Bhroum Sah Rahave Namaha",
        "gayatri_mantra": "Om Naagadhwajaya Vidmahe Padmahastaya Dheemahi Tanno Rahuh Prachodayat",
        "mantra_repetitions": "18,000 chants over 40 days or 108 times daily at dusk/night",
        "gemstone": {
            "name": "Hessonite Garnet (Gomedh)",
            "substitute": "Orange Zircon",
            "metal": "Silver or Ashtadhatu",
            "finger": "Middle finger of right hand",
            "wearing_time": "Saturday night or Wednesday night",
            "safety_caution": "Do not wear without verifying Rahu's placement. Can heighten phobias or erratic impulses if misprescribed.",
        },
        "charity_daan": {
            "items": "Seven mixed grains (Satnaja), dark blankets, blue cloth, or lead",
            "ideal_day": "Saturday or Wednesday night",
            "recipient": "Lepers, sweepers, or animal shelters",
        },
        "lifestyle_remedy": "Feed birds seven grains daily. Avoid alcohol, gambling, and clutter at the entrance of the home.",
        "classical_grounding": "Brihat Parashara Hora Shastra, Ch. 84; Saravali Ch. 30",
    },
    "ketu": {
        "planet": "Ketu",
        "beej_mantra": "Om Sraam Sreem Sroum Sah Ketave Namaha",
        "gayatri_mantra": "Om Ashwadhwajaya Vidmahe Shoolahastaya Dheemahi Tanno Ketuh Prachodayat",
        "mantra_repetitions": "17,000 chants over 40 days or 108 times daily at night",
        "gemstone": {
            "name": "Cat's Eye (Chrysoberyl / Vaidurya)",
            "substitute": "Tiger's Eye",
            "metal": "Silver or Panchadhatu",
            "finger": "Little finger or middle finger",
            "wearing_time": "Tuesday or Thursday night",
            "safety_caution": "Can cause sudden psychological or digestive upheaval if ill-placed. Strict chart confirmation required.",
        },
        "charity_daan": {
            "items": "Brown/spotted blankets, sesame seeds, mustard seeds, or sour food",
            "ideal_day": "Tuesday or Thursday",
            "recipient": "Spiritual mendicants, monks, or rescue animals",
        },
        "lifestyle_remedy": "Feed stray street dogs (especially black/brown dogs). Engage in silent meditation and spiritual detachment.",
        "classical_grounding": "Brihat Parashara Hora Shastra, Ch. 84",
    },
}


def calculate_dasha_periods(
    birth_date: str,
    birth_time: str = "12:00",
    moon_nakshatra: str = "",
    current_date: str = "",
) -> dict:
    """Calculates Vimshottari Dasha planetary periods and active timing windows.

    Evaluates the 120-year cycle to determine the current Mahadasha,
    Antardasha, start/end dates, and timing characteristics for events.

    Args:
        birth_date: Birth date in ISO format (YYYY-MM-DD).
        birth_time: Birth time (HH:MM) in 24h format, defaults to 12:00.
        moon_nakshatra: Optional birth lunar mansion (e.g. 'Rohini', 'Ashwini', 'Pushya').
            If omitted, estimated from the birth date.
        current_date: Target query date in ISO format (YYYY-MM-DD), defaults to current date.

    Returns:
        dict with status, calculated active Mahadasha, Antardasha, period duration,
        planetary rulers, timing quality (favorable/mixed/challenging), and themes.
    """
    try:
        b_dt = datetime.datetime.fromisoformat(birth_date)
    except Exception as e:
        return {"status": "error", "error": f"Invalid birth_date format '{birth_date}': {e}"}

    if current_date:
        try:
            c_dt = datetime.datetime.fromisoformat(current_date)
        except Exception as e:
            return {"status": "error", "error": f"Invalid current_date format '{current_date}': {e}"}
    else:
        c_dt = datetime.datetime.now(datetime.timezone.utc)

    # Determine starting Mahadasha lord from nakshatra or birth hash
    nakshatra_norm = moon_nakshatra.strip().lower()
    start_lord = _NAKSHATRA_TO_LORD.get(nakshatra_norm)
    if not start_lord:
        # If not matched or omitted, estimate deterministic starting planet from birth day of year
        day_index = (b_dt.timetuple().tm_yday + b_dt.year) % len(_VIMSHOTTARI_ORDER)
        start_lord = _VIMSHOTTARI_ORDER[day_index][0]

    # Find position in Vimshottari order
    order_names = [p[0] for p in _VIMSHOTTARI_ORDER]
    start_idx = order_names.index(start_lord)
    rotated_order = _VIMSHOTTARI_ORDER[start_idx:] + _VIMSHOTTARI_ORDER[:start_idx]

    # Calculate timeline in days
    total_age_days = (c_dt.date() - b_dt.date()).days
    if total_age_days < 0:
        return {"status": "error", "error": "current_date cannot be earlier than birth_date"}

    current_cursor_days = 0
    active_md = rotated_order[0][0]
    md_start_date = b_dt.date()
    md_end_date = b_dt.date()

    # Step through Mahadashas
    for p_name, p_years in rotated_order:
        p_days = int(p_years * 365.2425)
        if current_cursor_days + p_days > total_age_days:
            active_md = p_name
            md_start_date = b_dt.date() + datetime.timedelta(days=current_cursor_days)
            md_end_date = md_start_date + datetime.timedelta(days=p_days)
            break
        current_cursor_days += p_days

    # Calculate Antardashas within this Mahadasha
    # AD follows same sequence starting from the Mahadasha lord
    md_years = dict(_VIMSHOTTARI_ORDER)[active_md]
    md_idx = order_names.index(active_md)
    ad_order = _VIMSHOTTARI_ORDER[md_idx:] + _VIMSHOTTARI_ORDER[:md_idx]

    ad_cursor_date = md_start_date
    active_ad = ad_order[0][0]
    ad_start_date = md_start_date
    ad_end_date = md_start_date

    target_date = c_dt.date()
    for ad_name, ad_years in ad_order:
        # Sub-period length = MD_years * AD_years / 120 (in years)
        ad_fraction_years = (md_years * ad_years) / 120.0
        ad_days = int(ad_fraction_years * 365.2425)
        next_ad_date = ad_cursor_date + datetime.timedelta(days=ad_days)
        if ad_cursor_date <= target_date <= next_ad_date or target_date < next_ad_date:
            active_ad = ad_name
            ad_start_date = ad_cursor_date
            ad_end_date = next_ad_date
            break
        ad_cursor_date = next_ad_date

    # Determine timing quality
    favorable_planets = {"Jupiter", "Venus", "Moon", "Mercury"}
    challenging_planets = {"Saturn", "Rahu", "Ketu", "Mars"}

    if active_md in favorable_planets and active_ad in favorable_planets:
        timing_quality = "favorable"
    elif active_md in challenging_planets and active_ad in challenging_planets:
        timing_quality = "challenging"
    else:
        timing_quality = "mixed"

    md_info = _PLANET_THEMES.get(active_md, {})
    ad_info = _PLANET_THEMES.get(active_ad, {})

    return {
        "status": "success",
        "birth_date": b_dt.date().isoformat(),
        "query_date": target_date.isoformat(),
        "moon_nakshatra": moon_nakshatra or "Estimated from birth date",
        "cycle_system": "Vimshottari Dasha (120 years)",
        "active_mahadasha": {
            "planet": active_md,
            "duration_years": md_years,
            "period": f"{md_start_date.isoformat()} to {md_end_date.isoformat()}",
            "core_significations": md_info.get("significations", ""),
        },
        "active_antardasha": {
            "planet": active_ad,
            "period": f"{ad_start_date.isoformat()} to {ad_end_date.isoformat()}",
            "sub_significations": ad_info.get("significations", ""),
        },
        "timing_quality": timing_quality,
        "combined_influence": f"{active_md}-{active_ad} period activating {md_info.get('significations', '')} flavored by {ad_info.get('significations', '')}.",
        "favorable_actions": list(set(md_info.get("favorable_actions", []) + ad_info.get("favorable_actions", []))),
        "caution_areas": list(set(md_info.get("caution_areas", []) + ad_info.get("caution_areas", []))),
    }


def search_remedy_knowledge_base(
    planet: str,
    dosha_or_issue: str = "",
    remedy_type: str = "all",
) -> dict:
    """Searches the verified Vedic astrology remedy knowledge base.

    Returns grounded remedies including mantras, gemstones (with safety caveats),
    charitable giving (daan), and lifestyle upayas grounded in classical texts.

    Args:
        planet: Planet name (e.g. 'Saturn', 'Jupiter', 'Mars', 'Rahu', 'Ketu', 'Sun', 'Moon', 'Mercury', 'Venus').
        dosha_or_issue: Optional specific affliction or issue (e.g. 'Sade Sati', 'Manglik', 'Career obstacles').
        remedy_type: Filter by 'all', 'mantra', 'gemstone', 'daan', or 'lifestyle'.

    Returns:
        dict with status, planetary remedy details, safety warnings, and citations.
    """
    planet_key = planet.strip().lower()
    # Normalize common names
    aliases = {
        "surya": "sun", "ravi": "sun",
        "chandra": "moon", "soma": "moon",
        "mangal": "mars", "kuja": "mars", "angarak": "mars",
        "budha": "mercury", "budh": "mercury",
        "guru": "jupiter", "brihaspati": "jupiter",
        "shukra": "venus", "sukra": "venus",
        "shani": "saturn", "sani": "saturn",
    }
    planet_key = aliases.get(planet_key, planet_key)

    if planet_key not in _REMEDY_DATABASE:
        valid_planets = list(_REMEDY_DATABASE.keys())
        return {
            "status": "error",
            "error": f"Planet '{planet}' not found in knowledge base. Supported planets: {valid_planets}",
        }

    data = _REMEDY_DATABASE[planet_key]
    result: dict[str, Any] = {
        "status": "success",
        "planet": data["planet"],
        "dosha_or_issue_context": dosha_or_issue or "General planetary balance",
        "classical_grounding": data["classical_grounding"],
    }

    remedy_filter = remedy_type.strip().lower()
    if remedy_filter in ("all", "mantra"):
        result["mantras"] = {
            "beej_mantra": data["beej_mantra"],
            "gayatri_mantra": data["gayatri_mantra"],
            "repetitions": data["mantra_repetitions"],
            "guideline": "Mantras are universally safe, non-invasive sattvic remedies.",
        }

    if remedy_filter in ("all", "gemstone"):
        result["gemstone"] = data["gemstone"]

    if remedy_filter in ("all", "daan", "charity"):
        result["charity_daan"] = data["charity_daan"]

    if remedy_filter in ("all", "lifestyle"):
        result["lifestyle_remedy"] = data["lifestyle_remedy"]

    return result

