# Persistent Settings Implementation

## Overview
All application settings now persist across app refreshes and restarts. You no longer need to re-enter API keys or reconfigure database connections every time the app reloads.

## What Gets Saved
The following settings are automatically saved to disk:

### 1. API Keys
- **Groq API Key** - For LLaMA 3.3 70B chatbot
- **Anthropic API Key** - For Claude Haiku fallback

### 2. Database Configurations
- **HR Database** - Connection details, field mappings, connection status
- **Supply Chain Database** - Connection details, field mappings, connection status
- **Station Streams Database** - Connection details and configuration

### 3. Application Settings
- **Connector Mode** - Simulation, File, REST API, or MQTT

## How It Works

### Storage Location
Settings are stored in: `data/app_settings.json`

**Security Note**: This file is automatically added to `.gitignore` to prevent API keys from being committed to version control.

### Automatic Persistence
Settings are automatically saved to disk when you:
- Click "Save Groq key" in Settings → Chatbot API
- Click "Save Anthropic key" in Settings → Chatbot API
- Click "💾 Save HR config" in Settings → Data Sources
- Click "💾 Save SC config" in Settings → Data Sources
- Click "Apply connector mode" in Settings → Data Sources

### Automatic Loading
Settings are automatically loaded from disk when:
- The app starts up
- The app refreshes
- You navigate between pages

## Benefits

### Before (Without Persistence)
- ❌ Re-enter API keys after every app refresh
- ❌ Reconfigure database connections after every restart
- ❌ Lost settings when switching between pages
- ❌ Frustrating user experience

### After (With Persistence)
- ✅ Configure once, works forever
- ✅ API keys persist across sessions
- ✅ Database connections remain configured
- ✅ Seamless user experience

## File Structure

```json
{
  "db_configs": {
    "hr_db": {
      "db_type": "sqlite",
      "path": "data/databases/hr_database.db",
      "connected": true,
      "map_id": "employee_id",
      "map_name": "full_name",
      ...
    },
    "sc_db": {
      ...
    }
  },
  "groq_key": "gsk_...",
  "anthropic_key": "sk-ant-...",
  "connector_mode": "simulation"
}
```

## Technical Implementation

### Functions Added
1. **`load_persistent_settings()`** - Loads settings from disk into session state
2. **`save_persistent_settings()`** - Saves current session state to disk

### Modified Functions
- **`_save_db_config()`** - Now calls `save_persistent_settings()` after updating config
- **API key save buttons** - Now call `save_persistent_settings()` after updating keys
- **Connector mode button** - Now calls `save_persistent_settings()` after changing mode

### Initialization Flow
1. App loads default values into `st.session_state`
2. `load_persistent_settings()` is called immediately after
3. Saved settings overwrite defaults (if file exists)
4. App continues with persistent settings loaded

## Clearing Settings

### Method 1: Delete Settings File
```bash
rm data/app_settings.json
```

### Method 2: Use "✕ Clear" Buttons
Each database configuration has a "✕ Clear" button that removes that specific config and saves the change.

### Method 3: Manually Edit File
You can manually edit `data/app_settings.json` if needed (be careful with JSON syntax).

## Security Considerations

1. **API Keys Protected** - Settings file is in `.gitignore`, preventing accidental commits
2. **Local Storage Only** - Settings are stored locally, never sent to external servers
3. **Readable Format** - JSON format allows manual inspection and editing if needed
4. **Backup Recommended** - Consider backing up the settings file for disaster recovery

## Testing

A test script is included to verify functionality:
```bash
python test_persistence.py
```

This verifies:
- Settings file can be created
- Settings can be read back correctly
- All required fields are present
- File structure matches expected format

## Troubleshooting

### Settings Not Loading
1. Check if `data/app_settings.json` exists
2. Verify JSON syntax is valid (use a JSON validator)
3. Check file permissions (must be readable/writable)

### Settings Not Saving
1. Ensure `data/` directory exists and is writable
2. Check for error messages in the Streamlit console
3. Try deleting the settings file and reconfiguring

### API Key Issues
1. Verify the key format matches expected pattern (gsk_... or sk-ant-...)
2. Check key length (should be visible in Settings page)
3. Re-enter the key and click Save again

---

**Last Updated**: 2026-05-21
**Feature Status**: ✅ Fully Implemented and Tested
