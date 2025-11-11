import streamlit as st
import time
import pandas as pd

st.set_page_config(page_title="Pattern Matching Visualizer", layout="wide")

st.title("🔍 Pattern Matching Visualizer")
st.markdown("Interactive visualization of **KMP** and **Boyer-Moore** string matching algorithms")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Speed control
    st.subheader("Animation Speed")
    speed = st.slider(
        "Adjust auto-play speed:",
        min_value=0.1,
        max_value=2.0,
        value=0.5,
        step=0.1,
        format="%.1f sec",
        help="Time delay between steps during auto-play"
    )
    
    # Educational mode toggle
    educational_mode = st.checkbox(
        "📚 Educational Mode",
        value=False,
        help="Show detailed explanations at each step"
    )

def compute_lps_array(pattern):
    """Compute the Longest Proper Prefix which is also Suffix array for KMP"""
    m = len(pattern)
    lps = [0] * m
    length = 0
    i = 1
    steps = []
    
    steps.append({
        'i': 0,
        'length': 0,
        'lps': [0] + ['-'] * (m - 1),
        'description': 'Initialize: LPS[0] is always 0'
    })
    
    while i < m:
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            steps.append({
                'i': i,
                'length': length,
                'lps': lps.copy(),
                'description': f"Match! pattern[{i}]='{pattern[i]}' == pattern[{length-1}]='{pattern[length-1]}', LPS[{i}] = {length}"
            })
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
                steps.append({
                    'i': i,
                    'length': length,
                    'lps': lps.copy(),
                    'description': f"Mismatch! pattern[{i}]='{pattern[i]}' != pattern[{length}]='{pattern[length]}', backtrack to length = LPS[{length}] = {lps[length-1] if length > 0 else 0}"
                })
            else:
                lps[i] = 0
                steps.append({
                    'i': i,
                    'length': 0,
                    'lps': lps.copy(),
                    'description': f"No match and length=0, LPS[{i}] = 0"
                })
                i += 1
    
    return lps, steps

def kmp_search(text, pattern):
    """KMP pattern matching with step-by-step visualization data"""
    n = len(text)
    m = len(pattern)
    lps, _ = compute_lps_array(pattern)
    
    steps = []
    i = 0  # index for text
    j = 0  # index for pattern
    comparisons = 0
    
    while i < n:
        comparisons += 1
        if pattern[j] == text[i]:
            steps.append({
                'text_pos': i,
                'pattern_pos': j,
                'pattern_start': i - j,
                'match': True,
                'description': f"Match! text[{i}]='{text[i]}' == pattern[{j}]='{pattern[j]}'"
            })
            i += 1
            j += 1
        
        if j == m:
            steps.append({
                'text_pos': i - 1,
                'pattern_pos': j - 1,
                'pattern_start': i - j,
                'match': True,
                'description': f"✅ Pattern found at index {i - j}!",
                'found': True
            })
            j = lps[j - 1]
        elif i < n and pattern[j] != text[i]:
            steps.append({
                'text_pos': i,
                'pattern_pos': j,
                'pattern_start': i - j,
                'match': False,
                'description': f"Mismatch! text[{i}]='{text[i]}' != pattern[{j}]='{pattern[j]}'"
            })
            if j != 0:
                old_j = j
                j = lps[j - 1]
                steps.append({
                    'text_pos': i,
                    'pattern_pos': j,
                    'pattern_start': i - j,
                    'match': False,
                    'description': f"Shift pattern using LPS: j = LPS[{old_j - 1}] = {j}",
                    'shift': True
                })
            else:
                i += 1
    
    return steps, comparisons

def compute_bad_character_table(pattern):
    """Compute Bad Character table for Boyer-Moore"""
    m = len(pattern)
    bad_char = {}
    
    for i in range(m):
        bad_char[pattern[i]] = i
    
    return bad_char

def compute_good_suffix_table(pattern):
    """Compute Good Suffix table for Boyer-Moore"""
    m = len(pattern)
    good_suffix = [0] * m
    border_pos = [0] * (m + 1)
    
    # Initialize border positions
    i = m
    j = m + 1
    border_pos[i] = j
    
    while i > 0:
        while j <= m and pattern[i - 1] != pattern[j - 1]:
            if good_suffix[j - 1] == 0:
                good_suffix[j - 1] = j - i
            j = border_pos[j]
        i -= 1
        j -= 1
        border_pos[i] = j
    
    # Case 2: pattern shift
    j = border_pos[0]
    for i in range(m):
        if good_suffix[i] == 0:
            good_suffix[i] = j
        if i == j:
            j = border_pos[j]
    
    return good_suffix

def boyer_moore_search(text, pattern):
    """Boyer-Moore pattern matching with step-by-step visualization"""
    n = len(text)
    m = len(pattern)
    bad_char = compute_bad_character_table(pattern)
    good_suffix = compute_good_suffix_table(pattern)
    
    steps = []
    s = 0  # shift of the pattern with respect to text
    comparisons = 0
    
    while s <= n - m:
        j = m - 1
        
        while j >= 0:
            comparisons += 1
            if pattern[j] == text[s + j]:
                steps.append({
                    'text_pos': s + j,
                    'pattern_pos': j,
                    'pattern_start': s,
                    'match': True,
                    'description': f"Match! text[{s + j}]='{text[s + j]}' == pattern[{j}]='{pattern[j]}' (comparing right to left)"
                })
                j -= 1
            else:
                steps.append({
                    'text_pos': s + j,
                    'pattern_pos': j,
                    'pattern_start': s,
                    'match': False,
                    'description': f"Mismatch! text[{s + j}]='{text[s + j]}' != pattern[{j}]='{pattern[j]}'"
                })
                break
        
        if j < 0:
            steps.append({
                'text_pos': s + m - 1,
                'pattern_pos': 0,
                'pattern_start': s,
                'match': True,
                'description': f"✅ Pattern found at index {s}!",
                'found': True
            })
            s += good_suffix[0] if m > 0 else 1
        else:
            char = text[s + j]
            bad_char_shift = j - bad_char.get(char, -1)
            good_suffix_shift = good_suffix[j]
            shift = max(bad_char_shift, good_suffix_shift)
            
            steps.append({
                'text_pos': s + j,
                'pattern_pos': j,
                'pattern_start': s,
                'match': False,
                'description': f"Bad char shift: {bad_char_shift}, Good suffix shift: {good_suffix_shift}. Use max = {shift}",
                'shift': True
            })
            s += shift
    
    return steps, comparisons

def visualize_text_pattern(text, pattern, pattern_start, current_text_pos, current_pattern_pos, is_match):
    """Create HTML visualization of text and pattern alignment"""
    html = '<div style="font-family: monospace; font-size: 18px; line-height: 2;">'
    
    # Text line
    html += '<div>'
    for i, char in enumerate(text):
        if i == current_text_pos:
            color = '#28a745' if is_match else '#dc3545'
            html += f'<span style="background-color: {color}; color: white; padding: 2px 6px; border-radius: 3px;">{char}</span>'
        else:
            html += f'<span style="padding: 2px 6px;">{char}</span>'
    html += '</div>'
    
    # Pattern line
    html += '<div style="margin-top: 5px;">'
    for i in range(len(text)):
        if pattern_start <= i < pattern_start + len(pattern):
            pattern_idx = i - pattern_start
            char = pattern[pattern_idx]
            if pattern_idx == current_pattern_pos:
                color = '#28a745' if is_match else '#dc3545'
                html += f'<span style="background-color: {color}; color: white; padding: 2px 6px; border-radius: 3px;">{char}</span>'
            else:
                html += f'<span style="background-color: #e9ecef; padding: 2px 6px; border-radius: 3px;">{char}</span>'
        else:
            html += f'<span style="padding: 2px 6px;">&nbsp;</span>'
    html += '</div>'
    
    html += '</div>'
    return html

# Preset examples
PRESET_EXAMPLES = {
    "Custom": {"text": "ABABDABACDABABCABAB", "pattern": "ABABCABAB"},
    "DNA Sequence 1": {"text": "GATATATGCATATACGAATATGCATATACGA", "pattern": "ATATACGA"},
    "DNA Sequence 2": {"text": "GCTAGCTAGCTAGCTAGCTAGCT", "pattern": "TAGCT"},
    "DNA Repeat": {"text": "AAAAAAAAAGAAAAAAAAAG", "pattern": "AAAAG"},
    "Text Search": {"text": "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG", "pattern": "THE"},
    "Palindrome": {"text": "ABACABADABACABA", "pattern": "ABACABA"},
    "No Match": {"text": "HELLO WORLD THIS IS A TEST", "pattern": "XYZ"}
}

# Preset selector
st.markdown("### 📋 Quick Start Examples")

preset_choice = st.selectbox(
    "Choose a preset example or select 'Custom' to enter your own:",
    options=list(PRESET_EXAMPLES.keys()),
    index=0,
    key="preset_selector"
)

# Initialize default values in session state if they don't exist
if 'text_input' not in st.session_state:
    st.session_state.text_input = PRESET_EXAMPLES[preset_choice]["text"]
if 'pattern_input' not in st.session_state:
    st.session_state.pattern_input = PRESET_EXAMPLES[preset_choice]["pattern"]

# When preset changes, update the input values
if preset_choice != "Custom":
    st.session_state.text_input = PRESET_EXAMPLES[preset_choice]["text"]
    st.session_state.pattern_input = PRESET_EXAMPLES[preset_choice]["pattern"]

# Input section
col1, col2 = st.columns(2)
with col1:
    text = st.text_input(
        "📄 Text (T):", 
        max_chars=100,
        key="text_input"
    )
        
with col2:
    pattern = st.text_input(
        "🔎 Pattern (P):", 
        max_chars=50,
        key="pattern_input"
    )

if not text or not pattern:
    st.warning("⚠️ Please enter both text and pattern to visualize the algorithms.")
    st.stop()

if len(pattern) > len(text):
    st.error("❌ Pattern cannot be longer than text!")
    st.stop()

# Algorithm selection
st.markdown("---")
algorithm = st.radio("Select Algorithm:", ["KMP (Knuth-Morris-Pratt)", "Boyer-Moore"], horizontal=True)

st.markdown("---")

# KMP Algorithm
if algorithm == "KMP (Knuth-Morris-Pratt)":
    st.header("🔵 KMP Algorithm Visualization")
    
    # LPS Array Construction
    st.subheader("Step 1: LPS Array Construction")
    st.markdown("**LPS (Longest Proper Prefix which is also Suffix)** array helps avoid re-comparing characters.")
    
    lps, lps_steps = compute_lps_array(pattern)
    
    if 'lps_step' not in st.session_state:
        st.session_state.lps_step = 0
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("⏮️ Previous", key="lps_prev", disabled=st.session_state.lps_step == 0):
            st.session_state.lps_step = max(0, st.session_state.lps_step - 1)
            st.rerun()
    with col2:
        if st.button("⏭️ Next", key="lps_next", disabled=st.session_state.lps_step >= len(lps_steps) - 1):
            st.session_state.lps_step = min(len(lps_steps) - 1, st.session_state.lps_step + 1)
            st.rerun()
    with col3:
        if st.button("🔄 Reset", key="lps_reset"):
            st.session_state.lps_step = 0
            st.rerun()
    
    current_lps_step = lps_steps[st.session_state.lps_step]
    st.info(f"**Step {st.session_state.lps_step + 1}/{len(lps_steps)}:** {current_lps_step['description']}")
    
    # Display pattern and LPS array
    pattern_df = pd.DataFrame({
        'Index': list(range(len(pattern))),
        'Character': list(pattern),
        'LPS': [str(x) for x in current_lps_step['lps']]
    })
    st.table(pattern_df)
    
    st.markdown("---")
    
    # Pattern Matching
    st.subheader("Step 2: Pattern Matching")
    
    kmp_steps, kmp_comparisons = kmp_search(text, pattern)
    
    if 'kmp_step' not in st.session_state:
        st.session_state.kmp_step = 0
    
    # Auto-play state management
    if 'kmp_playing' not in st.session_state:
        st.session_state.kmp_playing = False
    
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        if st.button("⏮️ Previous", key="kmp_prev", disabled=st.session_state.kmp_step == 0):
            st.session_state.kmp_step = max(0, st.session_state.kmp_step - 1)
            st.session_state.kmp_playing = False
            st.rerun()
    with col2:
        if st.button("⏭️ Next", key="kmp_next", disabled=st.session_state.kmp_step >= len(kmp_steps) - 1):
            st.session_state.kmp_step = min(len(kmp_steps) - 1, st.session_state.kmp_step + 1)
            st.session_state.kmp_playing = False
            st.rerun()
    with col3:
        play_text = "⏸️ Pause" if st.session_state.kmp_playing else "⏩ Auto Play"
        if st.button(play_text, key="kmp_auto"):
            st.session_state.kmp_playing = not st.session_state.kmp_playing
            st.rerun()
    with col4:
        if st.button("🔄 Reset", key="kmp_reset"):
            st.session_state.kmp_step = 0
            st.session_state.kmp_playing = False
            st.rerun()
    
    # Handle auto-play
    if st.session_state.kmp_playing and st.session_state.kmp_step < len(kmp_steps) - 1:
        time.sleep(speed)
        st.session_state.kmp_step += 1
        st.rerun()
    elif st.session_state.kmp_playing and st.session_state.kmp_step >= len(kmp_steps) - 1:
        st.session_state.kmp_playing = False
    
    if kmp_steps:
        current_step = kmp_steps[st.session_state.kmp_step]
        
        st.info(f"**Step {st.session_state.kmp_step + 1}/{len(kmp_steps)}:** {current_step['description']}")
        
        # Educational mode explanation
        if educational_mode:
            with st.expander("📖 Understanding this step", expanded=True):
                if current_step.get('match'):
                    st.markdown("""
                    **What's happening:**
                    - The characters match, so we move both pointers forward
                    - KMP continues comparing from left to right
                    - If we complete the pattern, we found a match!
                    """)
                elif current_step.get('shift'):
                    st.markdown(f"""
                    **Why are we shifting?**
                    - A mismatch occurred, but we don't start over from the beginning
                    - The LPS array tells us how many characters we can skip
                    - This is KMP's key advantage: we never re-scan matched characters
                    - LPS value of {lps[st.session_state.kmp_step % len(lps)]} guides our next comparison position
                    """)
                else:
                    st.markdown("""
                    **KMP Strategy:**
                    - Compare characters left to right
                    - On match: advance both text and pattern pointers
                    - On mismatch: use LPS array to skip redundant comparisons
                    - This gives us O(n+m) time complexity
                    """)
        
        # Visualization
        html = visualize_text_pattern(
            text, 
            pattern, 
            current_step['pattern_start'],
            current_step['text_pos'],
            current_step['pattern_pos'],
            current_step['match']
        )
        st.markdown(html, unsafe_allow_html=True)
        
        # Statistics
        st.metric("Total Comparisons", kmp_comparisons)

# Boyer-Moore Algorithm
else:
    st.header("🔴 Boyer-Moore Algorithm Visualization")
    
    # Preprocessing Tables
    st.subheader("Step 1: Preprocessing Tables")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Bad Character Rule**")
        st.markdown("Shift based on last occurrence of mismatched character.")
        bad_char = compute_bad_character_table(pattern)
        
        bc_data = []
        unique_chars = sorted(set(pattern))
        for char in unique_chars:
            bc_data.append({
                'Character': char,
                'Last Position': bad_char.get(char, -1)
            })
        
        bc_df = pd.DataFrame(bc_data)
        st.table(bc_df)
    
    with col2:
        st.markdown("**Good Suffix Rule**")
        st.markdown("Shift based on matched suffix pattern.")
        good_suffix = compute_good_suffix_table(pattern)
        
        gs_data = []
        for i in range(len(pattern)):
            gs_data.append({
                'Position': i,
                'Character': pattern[i],
                'Shift': good_suffix[i]
            })
        
        gs_df = pd.DataFrame(gs_data)
        st.table(gs_df)
    
    st.markdown("---")
    
    # Pattern Matching
    st.subheader("Step 2: Pattern Matching (Right to Left)")
    
    bm_steps, bm_comparisons = boyer_moore_search(text, pattern)
    
    if 'bm_step' not in st.session_state:
        st.session_state.bm_step = 0
    
    # Auto-play state management
    if 'bm_playing' not in st.session_state:
        st.session_state.bm_playing = False
    
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        if st.button("⏮️ Previous", key="bm_prev", disabled=st.session_state.bm_step == 0):
            st.session_state.bm_step = max(0, st.session_state.bm_step - 1)
            st.session_state.bm_playing = False
            st.rerun()
    with col2:
        if st.button("⏭️ Next", key="bm_next", disabled=st.session_state.bm_step >= len(bm_steps) - 1):
            st.session_state.bm_step = min(len(bm_steps) - 1, st.session_state.bm_step + 1)
            st.session_state.bm_playing = False
            st.rerun()
    with col3:
        play_text = "⏸️ Pause" if st.session_state.bm_playing else "⏩ Auto Play"
        if st.button(play_text, key="bm_auto"):
            st.session_state.bm_playing = not st.session_state.bm_playing
            st.rerun()
    with col4:
        if st.button("🔄 Reset", key="bm_reset"):
            st.session_state.bm_step = 0
            st.session_state.bm_playing = False
            st.rerun()
    
    # Handle auto-play
    if st.session_state.bm_playing and st.session_state.bm_step < len(bm_steps) - 1:
        time.sleep(speed)
        st.session_state.bm_step += 1
        st.rerun()
    elif st.session_state.bm_playing and st.session_state.bm_step >= len(bm_steps) - 1:
        st.session_state.bm_playing = False
    
    if bm_steps:
        current_step = bm_steps[st.session_state.bm_step]
        
        st.info(f"**Step {st.session_state.bm_step + 1}/{len(bm_steps)}:** {current_step['description']}")
        
        # Educational mode explanation
        if educational_mode:
            with st.expander("📖 Understanding this step", expanded=True):
                if current_step.get('match') and not current_step.get('found'):
                    st.markdown("""
                    **What's happening:**
                    - Characters match! Boyer-Moore compares from RIGHT to LEFT
                    - We continue moving leftward in the pattern
                    - This right-to-left scanning is Boyer-Moore's unique approach
                    """)
                elif current_step.get('found'):
                    st.markdown("""
                    **Pattern Found! 🎉**
                    - All characters matched from right to left
                    - Boyer-Moore successfully found the pattern
                    - We can continue searching for more occurrences
                    """)
                elif current_step.get('shift'):
                    st.markdown("""
                    **Why this shift amount?**
                    - Boyer-Moore uses TWO heuristics: Bad Character and Good Suffix
                    - **Bad Character Rule**: Based on where the mismatched character last appeared
                    - **Good Suffix Rule**: Based on the matched suffix pattern
                    - We take the MAXIMUM of both shifts for optimal performance
                    - This allows larger jumps than KMP, especially in best cases!
                    """)
                else:
                    st.markdown("""
                    **Boyer-Moore Strategy:**
                    - Scan pattern from RIGHT to LEFT
                    - On mismatch: use Bad Character and Good Suffix rules
                    - Can skip many characters at once
                    - Best case: O(n/m) when pattern doesn't appear in text
                    """)
        
        # Visualization
        html = visualize_text_pattern(
            text, 
            pattern, 
            current_step['pattern_start'],
            current_step['text_pos'],
            current_step['pattern_pos'],
            current_step['match']
        )
        st.markdown(html, unsafe_allow_html=True)
        
        # Statistics
        st.metric("Total Comparisons", bm_comparisons)

# Footer with Complexity Comparison
st.markdown("---")
st.header("📊 Algorithm Complexity Comparison")

# Visual comparison chart
st.subheader("Time Complexity Comparison")

# Create comparison data for visualization
complexity_comparison = pd.DataFrame({
    'Case': ['Best Case', 'Average Case', 'Worst Case'],
    'KMP': [1.0, 1.0, 1.0],  # O(n+m) normalized
    'Boyer-Moore Best': [0.3, 0.7, 3.0]  # O(n/m) best, O(n) average, O(nm) worst (normalized for visualization)
})

# Rename for clarity
chart_data = pd.DataFrame({
    'Best Case': [1.0, 0.3],
    'Average Case': [1.0, 0.7],
    'Worst Case': [1.0, 3.0]
}, index=['KMP', 'Boyer-Moore'])

st.bar_chart(chart_data)
st.caption("Relative time complexity (lower is better). KMP baseline = 1.0")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("KMP Algorithm")
    
    complexity_data = {
        'Aspect': ['Time - Best', 'Time - Average', 'Time - Worst', 'Space', 'Preprocessing'],
        'Complexity': ['O(n+m)', 'O(n+m)', 'O(n+m)', 'O(m)', 'O(m)']
    }
    df = pd.DataFrame(complexity_data)
    st.dataframe(df, width='stretch', hide_index=True)
    
    st.markdown("""
    **Advantages:**
    - ✅ Consistent O(n+m) performance
    - ✅ Never re-scans text characters
    - ✅ Good for small alphabets
    - ✅ Simple preprocessing
    
    **Use Cases:**
    - DNA sequence matching
    - Small alphabet strings
    - When consistency matters
    """)

with col2:
    st.subheader("Boyer-Moore Algorithm")
    
    complexity_data = {
        'Aspect': ['Time - Best', 'Time - Average', 'Time - Worst', 'Space', 'Preprocessing'],
        'Complexity': ['O(n/m)', 'O(n)', 'O(nm)', 'O(m+σ)', 'O(m+σ)']
    }
    df = pd.DataFrame(complexity_data)
    st.dataframe(df, width='stretch', hide_index=True)
    
    st.markdown("""
    **Advantages:**
    - ✅ Very fast in practice
    - ✅ Larger jumps possible
    - ✅ Scans right-to-left
    - ✅ Excellent for large alphabets
    
    **Use Cases:**
    - Text editors (search/replace)
    - Large alphabet strings
    - When average case matters
    
    *σ = alphabet size*
    """)

st.markdown("---")
st.markdown("""
### 🎯 Key Differences
- **Scanning Direction**: KMP scans left-to-right, Boyer-Moore scans right-to-left
- **Shift Strategy**: KMP uses LPS array, Boyer-Moore uses Bad Character + Good Suffix rules
- **Performance**: KMP is consistent, Boyer-Moore can be faster with large alphabets
- **Best For**: KMP for small alphabets, Boyer-Moore for text searching
""")
