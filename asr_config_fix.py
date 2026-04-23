# Fix ASR configuration to use Whisper only for English, MMS for all others

# ORIGINAL configuration (in your notebook):
# LANG_ASR_CONFIG = {
#     'ben': ('mms', 'ben'),       # MMS for Bengali
#     'hin': ('mms', 'hin'),       # MMS for Hindi
#     'arb': ('mms', 'ara'),       # MMS for Arabic
#     'cmn': ('whisper', 'zh'),    # Whisper for Mandarin Chinese - CHANGE THIS
#     'eng': ('whisper', 'en'),    # Whisper for English
# }

# NEW configuration - MMS for all except English:
LANG_ASR_CONFIG = {
    'ben': ('mms', 'ben'),       # MMS for Bengali
    'hin': ('mms', 'hin'),       # MMS for Hindi  
    'arb': ('mms', 'ara'),       # MMS for Arabic
    'cmn': ('mms', 'cmn'),       # MMS for Mandarin Chinese (CHANGED)
    'eng': ('whisper', 'en'),    # Whisper for English only
}

# You also need to update the MMS_LANG_MAP to include Chinese:
MMS_LANG_MAP = {
    'ben': 'ben',  # Bengali
    'hin': 'hin',  # Hindi
    'arb': 'ara',  # Arabic (MMS uses 'ara' for Arabic)
    'cmn': 'cmn',  # Chinese Mandarin (ADD THIS)
}

print("Updated ASR configuration:")
print("- Whisper: English only")
print("- MMS: Bengali, Hindi, Arabic, Chinese")
print("")
print("Replace the LANG_ASR_CONFIG dictionary in your notebook with the new one above.")