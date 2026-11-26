    const PITCH_TYPES = ["FF", "SI", "SL", "CH", "FC", "ST", "CU", "FS", "KC"];
    const HISTORY_PITCH_TYPES = ["NONE", ...PITCH_TYPES, "OTHER"];

    const PITCH_LABELS = {
      NONE: "이전 투구 없음",
      FF: "포심 패스트볼",
      SI: "싱커",
      SL: "슬라이더",
      CH: "체인지업",
      FC: "커터",
      ST: "스위퍼",
      CU: "커브",
      FS: "스플리터",
      KC: "너클커브",
      OTHER: "기타 구종"
    };

    const PITCH_ENGLISH = {
      FF: "4-Seam Fastball",
      SI: "Sinker",
      SL: "Slider",
      CH: "Changeup",
      FC: "Cutter",
      ST: "Sweeper",
      CU: "Curveball",
      FS: "Splitter",
      KC: "Knuckle Curve",
      OTHER: "Other"
    };

    const PREV_LABELS = ["5구 전", "4구 전", "3구 전", "2구 전", "1구 전"];

    function pitchLabel(code) { return PITCH_LABELS[code] || code; }
    function pitchEnglish(code) { return PITCH_ENGLISH[code] || code; }
    function pitchOptionLabel(code) { return code === "NONE" ? "이전 투구 없음" : `${pitchLabel(code)} (${pitchEnglish(code)})`; }

    const ZONE_LAYOUT_PITCHER_VIEW = {
      Z12: { left: 0, top: 0, width: 50, height: 50, type: "outside", corner: "corner-tl" },
      Z11: { left: 50, top: 0, width: 50, height: 50, type: "outside", corner: "corner-tr" },
      Z14: { left: 0, top: 50, width: 50, height: 50, type: "outside", corner: "corner-bl" },
      Z13: { left: 50, top: 50, width: 50, height: 50, type: "outside", corner: "corner-br" },
      Z3: { left: 18, top: 18, width: 21.333, height: 21.333, type: "inside" },
      Z2: { left: 39.333, top: 18, width: 21.333, height: 21.333, type: "inside" },
      Z1: { left: 60.666, top: 18, width: 21.333, height: 21.333, type: "inside" },
      Z6: { left: 18, top: 39.333, width: 21.333, height: 21.333, type: "inside" },
      Z5: { left: 39.333, top: 39.333, width: 21.333, height: 21.333, type: "inside" },
      Z4: { left: 60.666, top: 39.333, width: 21.333, height: 21.333, type: "inside" },
      Z9: { left: 18, top: 60.666, width: 21.333, height: 21.333, type: "inside" },
      Z8: { left: 39.333, top: 60.666, width: 21.333, height: 21.333, type: "inside" },
      Z7: { left: 60.666, top: 60.666, width: 21.333, height: 21.333, type: "inside" }
    };

    const DEFAULT_PREV_ROWS = [
      { pitch: "FF", zone: "Z5" },
      { pitch: "SL", zone: "Z13" },
      { pitch: "FF", zone: "Z2" },
      { pitch: "CH", zone: "Z14" },
      { pitch: "SL", zone: "Z7" }
    ];

    let prevRows = JSON.parse(JSON.stringify(DEFAULT_PREV_ROWS));
    let activeTarget = { type: "prev", index: 0 };
    let compareZone = "Z13";
    let lastRecommendData = null;
    let requestVersion = 0;

    function createBatterSvg(side) {
      const mirrorClass = side === "L" ? "mirror" : "";
      return `
        <svg class="svg-figure ${mirrorClass}" viewBox="0 0 100 140" xmlns="http://www.w3.org/2000/svg" aria-label="batter">
          <path d="M45 92 L34 128" stroke="#334155" stroke-width="8" stroke-linecap="round"/>
          <path d="M56 92 L70 128" stroke="#334155" stroke-width="8" stroke-linecap="round"/>
          <path d="M34 48 Q50 38 66 48 L70 86 Q50 101 30 86 Z" fill="#fca5a5" stroke="#b91c1c" stroke-width="3"/>
          <path d="M50 43 L50 96" stroke="#fee2e2" stroke-width="4" opacity="0.9"/>
          <circle cx="50" cy="28" r="16" fill="#f7d7ad" stroke="#92400e" stroke-width="3"/>
          <path d="M31 27 Q35 10 55 11 Q70 13 75 28 L55 28 Q49 22 37 28 Z" fill="#991b1b" stroke="#7f1d1d" stroke-width="2"/>
          <path d="M62 54 Q76 60 82 78" stroke="#7f1d1d" stroke-width="8" stroke-linecap="round"/>
          <path d="M39 54 Q28 63 24 77" stroke="#7f1d1d" stroke-width="7" stroke-linecap="round"/>
          <path d="M78 18 L96 96" stroke="#78350f" stroke-width="7" stroke-linecap="round"/>
          <path d="M78 18 L96 96" stroke="#f59e0b" stroke-width="2" stroke-linecap="round" opacity="0.55"/>
          <circle cx="77" cy="55" r="6" fill="#f7d7ad" stroke="#92400e" stroke-width="2"/>
          <circle cx="80" cy="63" r="6" fill="#f7d7ad" stroke="#92400e" stroke-width="2"/>
          <ellipse cx="32" cy="130" rx="12" ry="5" fill="#1f2937"/>
          <ellipse cx="72" cy="130" rx="12" ry="5" fill="#1f2937"/>
        </svg>
      `;
    }

    function createPitcherSvg(hand) {
      const mirrorClass = hand === "L" ? "mirror" : "";
      return `
        <svg class="svg-figure ${mirrorClass}" viewBox="0 0 100 140" xmlns="http://www.w3.org/2000/svg" aria-label="pitcher">
          <path d="M45 92 L35 128" stroke="#334155" stroke-width="8" stroke-linecap="round"/>
          <path d="M55 92 L78 122" stroke="#334155" stroke-width="8" stroke-linecap="round"/>
          <path d="M34 48 Q50 38 66 48 L70 86 Q50 101 30 86 Z" fill="#93c5fd" stroke="#1d4ed8" stroke-width="3"/>
          <path d="M50 43 L50 96" stroke="#dbeafe" stroke-width="4" opacity="0.9"/>
          <circle cx="50" cy="28" r="16" fill="#f7d7ad" stroke="#92400e" stroke-width="3"/>
          <path d="M31 27 Q35 10 55 11 Q70 13 75 28 L56 28 Q48 22 37 28 Z" fill="#1d4ed8" stroke="#1e3a8a" stroke-width="2"/>
          <path d="M58 21 L80 24" stroke="#1e3a8a" stroke-width="5" stroke-linecap="round"/>
          <path d="M38 55 Q22 62 15 82" stroke="#1e293b" stroke-width="8" stroke-linecap="round"/>
          <ellipse cx="13" cy="85" rx="10" ry="13" fill="#78350f" stroke="#451a03" stroke-width="2"/>
          <path d="M63 53 Q82 43 88 26" stroke="#1e293b" stroke-width="8" stroke-linecap="round"/>
          <circle cx="90" cy="22" r="5" fill="#ffffff" stroke="#94a3b8" stroke-width="2"/>
          <ellipse cx="34" cy="130" rx="12" ry="5" fill="#1f2937"/>
          <ellipse cx="80" cy="124" rx="12" ry="5" fill="#1f2937"/>
        </svg>
      `;
    }

    function showMessage(text, type = "ok") {
      const el = document.getElementById("message");
      el.textContent = text;
      el.className = `message show ${type}`;
    }

    function showResultMessage(text, warnings = []) {
      showMessage([text, ...warnings].join(" "), warnings.length ? "warn" : "ok");
    }

    function clearMessage() {
      const el = document.getElementById("message");
      el.textContent = "";
      el.className = "message";
    }

    const FIELD_RULES = [
      { id: "balls", label: "볼", min: 0, max: 3 },
      { id: "strikes", label: "스트라이크", min: 0, max: 2 },
      { id: "outs_when_up", label: "아웃카운트", min: 0, max: 2 },
      { id: "inning", label: "이닝", min: 1, max: 20 },
      { id: "top_n", label: "최종 추천 개수", min: 1, max: 20 }
    ];

    const FIELD_LABELS = {
      balls: "볼",
      strikes: "스트라이크",
      outs_when_up: "아웃카운트",
      inning: "이닝",
      top_n: "최종 추천 개수",
      runner_1b: "1루 주자",
      runner_2b: "2루 주자",
      runner_3b: "3루 주자",
      stand: "타자 방향",
      p_throws: "투수 손"
    };

    function validateInputs() {
      const errors = [];

      FIELD_RULES.forEach(rule => {
        const el = document.getElementById(rule.id);
        const raw = String(el.value || "").trim();
        const value = Number(raw);

        if (raw === "" || !Number.isInteger(value)) {
          errors.push(`${rule.label}은 정수로 입력해 주세요.`);
          return;
        }

        if (value < rule.min || value > rule.max) {
          errors.push(`${rule.label}은 ${rule.min}–${rule.max} 범위로 입력해 주세요.`);
        }
      });

      if (manualCandidates() && getCandidatePitchTypes().length === 0) {
        errors.push("후보 구종을 최소 1개 이상 선택해 주세요.");
      }

      if (errors.length > 0) {
        showMessage(errors.join(" "), "error");
        return false;
      }

      return true;
    }

    function formatApiError(data) {
      if (!data) return "요청 처리 중 오류가 발생했습니다.";

      if (Array.isArray(data.detail)) {
        return data.detail.map(err => {
          if (err.type === "pitch_history_length_mismatch" && typeof err.msg === "string") {
            return err.msg;
          }

          const field = Array.isArray(err.loc) ? err.loc[err.loc.length - 1] : "";
          const label = FIELD_LABELS[field] || field || "입력값";
          const ctx = err.ctx || {};

          if (err.type === "less_than_equal" && ctx.le !== undefined) {
            return `${label}은 최대 ${ctx.le}까지 입력할 수 있습니다.`;
          }

          if (err.type === "greater_than_equal" && ctx.ge !== undefined) {
            return `${label}은 최소 ${ctx.ge} 이상이어야 합니다.`;
          }

          return `${label} 입력값을 확인해 주세요.`;
        }).join(" ");
      }

      if (typeof data.detail === "string") return data.detail;
      if (typeof data.error === "string") return data.error;

      return "입력값을 확인해 주세요.";
    }

    function getCandidatePitchTypes() {
      const checks = document.querySelectorAll("#candidatePitchChecks input[type='checkbox']:checked");
      return Array.from(checks).map(x => x.value);
    }

    function manualCandidates() {
      return document.getElementById("candidateMode").value === "manual";
    }

    function makePitchOptions(selected) {
      return HISTORY_PITCH_TYPES.map(p => `<option value="${p}" ${p === selected ? "selected" : ""}>${pitchOptionLabel(p)}</option>`).join("");
    }

    function renderCandidatePitchChecks() {
      const box = document.getElementById("candidatePitchChecks");
      box.innerHTML = "";
      const defaultChecked = new Set(["FF", "SL", "CH"]);
      PITCH_TYPES.forEach(p => {
        const label = document.createElement("label");
        label.className = "pitch-check";
        label.innerHTML = `
          <input type="checkbox" value="${p}" ${defaultChecked.has(p) ? "checked" : ""} />
          ${pitchLabel(p)} (${pitchEnglish(p)})
        `;
        box.appendChild(label);
      });
      updateCandidateMode();
    }

    function updateCandidateMode() {
      const manual = manualCandidates();
      document.querySelectorAll('#candidatePitchChecks input').forEach(input => { input.disabled = !manual; });
    }

    function riskClass(value) {
      if (value < -0.01) return "score-good";
      if (value > 0.02) return "score-bad";
      return "score-neutral";
    }

    function sourceCountLabel(item) {
      if (item.source === "global" && item.source_count === 0) return "미제공";
      return item.source_count.toLocaleString();
    }

    function sourceLabel(source) {
      const labels = {
        "pitcher+count+action+zone": "같은 투수·카운트·구종·위치 기준",
        "pitcher+action+zone": "같은 투수·구종·위치 기준",
        "count+action+zone": "같은 카운트·구종·위치 기준",
        "action+zone": "같은 구종·위치 기준",
        "action": "같은 구종 기준",
        "global": "전체 평균 기준"
      };
      return labels[source] || source;
    }

    function getBestRecommendedZone() {
      if (!lastRecommendData || !lastRecommendData.recommendations || lastRecommendData.recommendations.length === 0) return null;
      return lastRecommendData.recommendations[0].zone;
    }

    function getActiveZone() {
      if (activeTarget.type === "prev") return prevRows[activeTarget.index].zone;
      if (activeTarget.type === "compare") return compareZone;
      return null;
    }

    function zoneCellClass(zone) {
      const classes = ["zone-cell"];
      const bestZone = getBestRecommendedZone();
      const activeZone = getActiveZone();
      const isRecommended = bestZone === zone;
      const isCompare = compareZone === zone && activeTarget.type === "compare";
      const isActive = activeZone === zone;
      if (isRecommended && isCompare) classes.push("both");
      else {
        if (isRecommended) classes.push("recommended");
        if (isCompare) classes.push("compare");
      }
      if (isActive) classes.push("active");
      return classes.join(" ");
    }

    function zoneSubLabel(zone) {
      return ZONE_LAYOUT_PITCHER_VIEW[zone].type === "inside" ? "존 안쪽" : "존 바깥쪽";
    }

    function renderZoneBoard(focusZone) {
      const board = document.getElementById("zoneBoard");
      const previousFocus = board.contains(document.activeElement) ? document.activeElement.dataset.zone : null;
      const zoneToFocus = focusZone ?? previousFocus;
      const activeZone = getActiveZone();
      let nextFocus;
      board.innerHTML = "";
      Object.entries(ZONE_LAYOUT_PITCHER_VIEW).forEach(([zone, cfg]) => {
        const button = document.createElement("button");
        button.type = "button";
        button.dataset.zone = zone;
        button.setAttribute("aria-pressed", String(zone === activeZone));
        button.className = [zoneCellClass(zone), cfg.type, cfg.corner || ""].join(" ").trim();
        button.style.left = `${cfg.left}%`;
        button.style.top = `${cfg.top}%`;
        button.style.width = `${cfg.width}%`;
        button.style.height = `${cfg.height}%`;
        button.onclick = () => selectZone(zone);
        button.innerHTML = `<span class="zone-label">${zone}</span><span class="zone-sub">${zoneSubLabel(zone)}</span>`;
        board.appendChild(button);
        if (zone === zoneToFocus) nextFocus = button;
      });
      updateActiveTargetText();
      nextFocus?.focus({ preventScroll: focusZone == null });
    }

    function updateActiveTargetText() {
      const el = document.getElementById("activeTargetText");
      if (activeTarget.type === "prev") el.textContent = `선택 대상: ${PREV_LABELS[activeTarget.index]} 위치`;
      else if (activeTarget.type === "compare") el.textContent = "선택 대상: 직접 비교 위치";
      else el.textContent = "선택 대상 없음";
    }

    function selectZone(zone) {
      requestVersion += 1;
      if (activeTarget.type === "prev") {
        if (prevRows[activeTarget.index].pitch === "NONE") {
          showMessage("이전 투구 없음 항목에는 위치를 지정할 수 없습니다.", "error");
          return;
        }
        prevRows[activeTarget.index].zone = zone;
        renderPrevRows();
      }
      if (activeTarget.type === "compare") {
        compareZone = zone;
        document.getElementById("compareZoneChip").textContent = zone;
      }
      renderZoneBoard();
    }

    function setActivePrev(index, moveFocus) {
      if (prevRows[index].pitch === "NONE") {
        showMessage("이전 투구 없음으로 설정된 항목은 위치를 선택할 필요가 없습니다.", "error");
        return;
      }
      activeTarget = { type: "prev", index };
      renderZoneBoard(moveFocus ? getActiveZone() : undefined);
      renderPrevRows();
    }

    function setActiveCompare(moveFocus) {
      activeTarget = { type: "compare" };
      renderZoneBoard(moveFocus ? getActiveZone() : undefined);
    }

    function renderPrevRows() {
      const container = document.getElementById("prevPitchRows");
      container.innerHTML = "";
      prevRows.forEach((row, idx) => {
        const div = document.createElement("div");
        div.className = "pitch-row";
        const isActive = activeTarget.type === "prev" && activeTarget.index === idx;
        const isNone = row.pitch === "NONE";
        div.innerHTML = `
          <div class="idx">${PREV_LABELS[idx]}</div>
          <select aria-label="${PREV_LABELS[idx]} 구종" onchange="updatePrevPitch(${idx}, this.value)">${makePitchOptions(row.pitch)}</select>
          <div class="zone-chip">${isNone ? "-" : row.zone}</div>
          <button class="${isActive ? "btn-primary" : "btn-ghost"}" onclick="setActivePrev(${idx}, event.detail === 0)" ${isNone ? "disabled" : ""}>위치 선택</button>
        `;
        container.appendChild(div);
      });
    }

    function updatePrevPitch(index, value) {
      prevRows[index].pitch = value;
      if (value === "NONE") prevRows[index].zone = null;
      else if (!prevRows[index].zone) prevRows[index].zone = "Z5";
      renderPrevRows();
      renderZoneBoard();
    }

    function renderComparePitchOptions() {
      const select = document.getElementById("compare_pitch_type");
      const options = PITCH_TYPES;
      const prevValue = select.value;

      select.innerHTML = options.map(p => `<option value="${p}">${pitchLabel(p)} (${pitchEnglish(p)})</option>`).join("");

      if (options.includes(prevValue)) select.value = prevValue;
      else select.value = options.includes("SL") ? "SL" : options[0];
    }

    function updateHandVisuals() {
      const stand = document.getElementById("stand").value;
      const pThrows = document.getElementById("p_throws").value;
      document.getElementById("leftBatterSlot").classList.toggle("active", stand === "L");
      document.getElementById("rightBatterSlot").classList.toggle("active", stand === "R");
      document.getElementById("leftBatterSvg").innerHTML = createBatterSvg("L");
      document.getElementById("rightBatterSvg").innerHTML = createBatterSvg("R");
      document.getElementById("leftPitcherSvg").innerHTML = createPitcherSvg("L");
      document.getElementById("rightPitcherSvg").innerHTML = createPitcherSvg("R");
      document.getElementById("leftPitcherSlot").classList.toggle("active", pThrows === "L");
      document.getElementById("rightPitcherSlot").classList.toggle("active", pThrows === "R");
      document.getElementById("pitcherDesc").textContent = pThrows === "L" ? "좌투수" : "우투수";
    }

    function buildPayload() {
      const candidateInput = manualCandidates() ? getCandidatePitchTypes() : null;
      const validPrevRows = prevRows.filter(row => row.pitch && row.pitch !== "NONE" && row.zone);
      return {
        pitcher: Number(document.getElementById("pitcher").value),
        batter: null,
        balls: Number(document.getElementById("balls").value),
        strikes: Number(document.getElementById("strikes").value),
        outs_when_up: Number(document.getElementById("outs_when_up").value),
        inning: Number(document.getElementById("inning").value),
        runner_1b: Number(document.getElementById("runner_1b").value),
        runner_2b: Number(document.getElementById("runner_2b").value),
        runner_3b: Number(document.getElementById("runner_3b").value),
        stand: document.getElementById("stand").value,
        p_throws: document.getElementById("p_throws").value,
        prev_pitch_types: validPrevRows.map(x => x.pitch),
        prev_zones: validPrevRows.map(x => x.zone),
        candidate_pitch_types: candidateInput,
        top_n: Number(document.getElementById("top_n").value)
      };
    }

    async function recommend() {
      const version = ++requestVersion;
      clearMessage();
      if (!validateInputs()) return;

      const payload = buildPayload();

      try {
        const res = await fetch("recommend", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
        const data = await res.json();
        if (version !== requestVersion) return;

        if (!res.ok) {
          showMessage(formatApiError(data), "error");
          return;
        }

        lastRecommendData = data;
        renderLstmTable(data.lstm_top3 || []);
        renderRecommendationTable(data.recommendations || []);
        renderZoneBoard();
        document.getElementById("rawJson").textContent = JSON.stringify(data, null, 2);
        showResultMessage("추천 결과를 생성했습니다.", data.warnings);
      } catch (err) {
        if (version !== requestVersion) return;
        showMessage("서버와 통신하는 중 오류가 발생했습니다. FastAPI 서버가 실행 중인지 확인해 주세요.", "error");
      }
    }

    async function compareAction() {
      const version = ++requestVersion;
      clearMessage();
      if (!validateInputs()) return;

      const payload = buildPayload();
      payload.action_pitch_type = document.getElementById("compare_pitch_type").value;
      payload.action_zone = compareZone;

      if (!payload.action_pitch_type) {
        showMessage("비교할 구종을 먼저 선택해 주세요.", "error");
        return;
      }

      try {
        const res = await fetch("compare", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
        const data = await res.json();
        if (version !== requestVersion) return;

        if (!res.ok || data.error) {
          showMessage(formatApiError(data), "error");
          return;
        }

        renderComparison(data);
        renderZoneBoard();
        document.getElementById("rawJson").textContent = JSON.stringify(data, null, 2);
        showResultMessage("직접 선택 비교를 완료했습니다.", data.warnings);
      } catch (err) {
        if (version !== requestVersion) return;
        showMessage("서버와 통신하는 중 오류가 발생했습니다. FastAPI 서버가 실행 중인지 확인해 주세요.", "error");
      }
    }

    function renderLstmTable(items) {
      const tbody = document.querySelector("#lstmTable tbody");
      tbody.innerHTML = "";
      if (!items.length) { tbody.innerHTML = `<tr><td colspan="3">결과 없음</td></tr>`; return; }
      items.forEach((item, idx) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `<td>${idx + 1}</td><td><span class="pitch-pill">${pitchLabel(item.pitch_type)}</span><span class="code-sub">${pitchEnglish(item.pitch_type)}</span></td><td>${(item.probability * 100).toFixed(2)}%</td>`;
        tbody.appendChild(tr);
      });
    }

    function renderRecommendationTable(items) {
      const tbody = document.querySelector("#recTable tbody");
      tbody.innerHTML = "";
      if (!items.length) { tbody.innerHTML = `<tr><td colspan="6">결과 없음</td></tr>`; return; }
      items.forEach((item, idx) => {
        const tr = document.createElement("tr");
        const cls = riskClass(item.pred_delta_run_exp);
        tr.innerHTML = `
          <td>${idx + 1}</td>
          <td><span class="pitch-pill">${pitchLabel(item.pitch_type)}</span><span class="code-sub">${pitchEnglish(item.pitch_type)}</span></td>
          <td><b>${item.zone}</b></td>
          <td class="${cls}">${item.pred_delta_run_exp.toFixed(5)}</td>
          <td>${sourceLabel(item.source)}</td>
          <td>${sourceCountLabel(item)}</td>
        `;
        tbody.appendChild(tr);
      });
    }

    function renderComparison(data) {
      const box = document.getElementById("compareResult");
      box.style.display = "grid";
      const selected = data.selected_action;
      const best = data.best_recommendation;
      const cmp = data.comparison;
      if (!best) { box.innerHTML = `<div class="mini-card">모델 추천 결과가 없습니다.</div>`; return; }
      const selectedCls = riskClass(selected.pred_delta_run_exp);
      const bestCls = riskClass(best.pred_delta_run_exp);
      const message = cmp.is_selected_better_or_equal ? "직접 선택 점수가 추천 1위와 같거나 낮습니다." : "추천 1위의 점수가 더 낮습니다.";
      box.innerHTML = `
        <div class="mini-card">
          <h4>직접 선택</h4>
          <div><span class="pitch-pill">${pitchLabel(selected.pitch_type)}</span><span class="code-sub">${pitchEnglish(selected.pitch_type)}</span> / <b>${selected.zone}</b></div>
          <div class="mini-value ${selectedCls}">${selected.pred_delta_run_exp.toFixed(5)}</div>
          <div style="color:#6b7280;font-size:12px;">${sourceLabel(selected.source)} · 표본 수 ${sourceCountLabel(selected)}</div>
        </div>
        <div class="mini-card">
          <h4>모델 추천 1위</h4>
          <div><span class="pitch-pill">${pitchLabel(best.pitch_type)}</span><span class="code-sub">${pitchEnglish(best.pitch_type)}</span> / <b>${best.zone}</b></div>
          <div class="mini-value ${bestCls}">${best.pred_delta_run_exp.toFixed(5)}</div>
          <div style="color:#6b7280;font-size:12px;">${sourceLabel(best.source)} · 표본 수 ${sourceCountLabel(best)}</div>
        </div>
        <div class="mini-card" style="grid-column:1 / -1;">
          <h4>모델 점수 비교</h4>
          <div>직접 선택 점수 − 추천 1위 점수 = <b class="${cmp.selected_minus_best <= 0 ? "score-good" : "score-bad"}">${cmp.selected_minus_best.toFixed(5)}</b></div>
          <div style="margin-top:6px;color:#475569;">${message}</div>
        </div>
      `;
    }

    async function loadHealth() {
      const healthText = document.getElementById("healthText");
      try {
        const res = await fetch("health");
        const data = await res.json();
        if (res.status === 503 && data.status === "not_ready") {
          healthText.textContent = data.reason === "no_supported_actions"
            ? "모델에 추천할 수 있는 구종 또는 위치가 없습니다."
            : "서빙 모델 파일을 준비한 뒤 서버를 재시작하세요.";
          return;
        }
        if (!res.ok || data.status !== "ok") {
          healthText.textContent = "서버 상태 확인 실패";
          return;
        }
        if (data.data_source === "synthetic") {
          document.getElementById("syntheticNotice").hidden = false;
          document.getElementById("pitcher").value = data.example_pitcher;
        }
        const pitchCount = (data.supported_pitch_types || []).length;
        healthText.innerHTML = `서버 정상 작동<br/>추천 엔진 준비 완료<br/>지원 구종 ${pitchCount}종`;
        document.getElementById("recommendButton").disabled = false;
        document.getElementById("compareButton").disabled = false;
      } catch (err) {
        healthText.textContent = "서버 상태 확인 실패";
      }
    }

    function resetDemo() {
      requestVersion += 1;
      lastRecommendData = null;
      renderLstmTable([]);
      renderRecommendationTable([]);
      const comparison = document.getElementById("compareResult");
      comparison.replaceChildren();
      comparison.style.display = "none";
      document.getElementById("rawJson").textContent = "아직 결과가 없습니다.";
      document.getElementById("balls").value = "1";
      document.getElementById("strikes").value = "2";
      document.getElementById("outs_when_up").value = "1";
      document.getElementById("inning").value = "7";
      document.getElementById("runner_1b").value = "0";
      document.getElementById("runner_2b").value = "1";
      document.getElementById("runner_3b").value = "0";
      document.getElementById("stand").value = "R";
      document.getElementById("p_throws").value = "R";
      document.getElementById("top_n").value = "5";
      document.getElementById("candidateMode").value = "manual";
      prevRows = JSON.parse(JSON.stringify(DEFAULT_PREV_ROWS));
      activeTarget = { type: "prev", index: 0 };
      compareZone = "Z13";
      document.getElementById("compareZoneChip").textContent = compareZone;
      renderCandidatePitchChecks();
      renderPrevRows();
      renderComparePitchOptions();
      renderZoneBoard();
      updateHandVisuals();
      clearMessage();
    }

    window.addEventListener("DOMContentLoaded", () => {
      document.querySelector("main").addEventListener("input", () => { requestVersion += 1; });
      renderCandidatePitchChecks();
      renderPrevRows();
      renderComparePitchOptions();
      renderZoneBoard();
      updateHandVisuals();
      loadHealth();
      document.getElementById("stand").addEventListener("change", updateHandVisuals);
      document.getElementById("p_throws").addEventListener("change", updateHandVisuals);
    });
