from __future__ import annotations

import streamlit as st


st.set_page_config(page_title="Synthetic Interactive Cockpit", layout="wide")


st.markdown(
    """
<style>
.cockpit-band {
    border: 1px solid #D8E0EA;
    border-radius: 14px;
    padding: 1rem 1.1rem;
    background: #FFFFFF;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
}
.cockpit-label {
    color: #64748B;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.cockpit-value {
    color: #0F172A;
    font-size: 1.2rem;
    font-weight: 800;
    margin-top: 0.2rem;
}
.cockpit-note {
    color: #475569;
    font-size: 0.88rem;
    line-height: 1.35;
    margin-top: 0.45rem;
}
.status-high { border-left: 5px solid #DC2626; }
.status-watch { border-left: 5px solid #D97706; }
.status-ready { border-left: 5px solid #059669; }
</style>
""",
    unsafe_allow_html=True,
)


st.title("Synthetic Interactive Cockpit")
st.caption(
    "Public demo data only. No employer data, real station figures, internal screenshots, "
    "private action routes, or operational evidence are shown."
)

st.info(
    "This cockpit preview demonstrates the method: expose timing risk, locate the handoff layer, "
    "route action ownership, and keep KPI evidence governed. Values and labels are synthetic."
)

top1, top2, top3 = st.columns(3)
with top1:
    st.markdown(
        """
<div class="cockpit-band status-watch">
  <div class="cockpit-label">TAT Timing-Risk Exposure</div>
  <div class="cockpit-value">Watch</div>
  <div class="cockpit-note">Turnaround pressure is treated as a decision-support signal before it becomes a departure outcome.</div>
</div>
""",
        unsafe_allow_html=True,
    )
with top2:
    st.markdown(
        """
<div class="cockpit-band status-high">
  <div class="cockpit-label">Handoff Ownership</div>
  <div class="cockpit-value">Decision Required</div>
  <div class="cockpit-note">The cockpit separates facts, assumptions, and accountable owner lanes at the handoff point.</div>
</div>
""",
        unsafe_allow_html=True,
    )
with top3:
    st.markdown(
        """
<div class="cockpit-band status-ready">
  <div class="cockpit-label">Validation Status</div>
  <div class="cockpit-value">Synthetic Manifest Ready</div>
  <div class="cockpit-note">Public demo artifacts are checked before KPI views, action lanes, or memo outputs are presented.</div>
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown("### Decision Lanes")
lane1, lane2, lane3 = st.columns(3)
with lane1:
    st.markdown(
        """
<div class="cockpit-band">
  <div class="cockpit-label">Ground Ops Lane</div>
  <div class="cockpit-value">Stabilize controllable flow</div>
  <div class="cockpit-note">Use synthetic delay-category evidence to route operational follow-up without exposing real station performance.</div>
</div>
""",
        unsafe_allow_html=True,
    )
with lane2:
    st.markdown(
        """
<div class="cockpit-band">
  <div class="cockpit-label">OCC / Hub Control Lane</div>
  <div class="cockpit-value">Protect the handoff</div>
  <div class="cockpit-note">Surface assumption gaps and escalation triggers before accountability becomes ambiguous.</div>
</div>
""",
        unsafe_allow_html=True,
    )
with lane3:
    st.markdown(
        """
<div class="cockpit-band">
  <div class="cockpit-label">Leadership Lane</div>
  <div class="cockpit-value">Decide the review focus</div>
  <div class="cockpit-note">Convert validated signals into a short executive memo, action lane, and next review question.</div>
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown("### Method Stack")
stack1, stack2, stack3, stack4 = st.columns(4)
for col, label, value in [
    (stack1, "Evidence", "Synthetic KPI contract"),
    (stack2, "Risk Lens", "TAT timing exposure"),
    (stack3, "Ownership", "DelayCategory routing"),
    (stack4, "Output", "Executive decision memo"),
]:
    with col:
        st.markdown(
            f"""
<div class="cockpit-band">
  <div class="cockpit-label">{label}</div>
  <div class="cockpit-value">{value}</div>
</div>
""",
            unsafe_allow_html=True,
        )
