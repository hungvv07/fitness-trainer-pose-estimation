// Object counting page: people, vehicles, everyday objects, line crossing
document.addEventListener('DOMContentLoaded', () => {
    const $ = (id) => document.getElementById(id);

    const feed = $('feed');
    const stage = $('stage');
    const placeholder = $('placeholder');
    const placeholderText = $('placeholder-text');
    const stageHint = $('stage-hint');
    const fileRow = $('file-row');
    const fileInput = $('file-input');
    const fileName = $('file-name');
    const startBtn = $('start-btn');
    const stopBtn = $('stop-btn');
    const resetBtn = $('reset-btn');
    const progressWrap = $('progress-wrap');
    const progressFill = $('progress-fill');
    const progressText = $('progress-text');
    const stateDot = $('state-dot');
    const stateText = $('state-text');
    const metrics = $('metrics');
    const lineOpts = $('line-opts');
    const linePos = $('line-pos');
    const linePosVal = $('line-pos-val');
    const conf = $('conf');
    const confVal = $('conf-val');
    const modelSelect = $('model-select');
    const groupCards = $('group-cards');
    const labelRows = $('label-rows');
    const labelTable = labelRows.closest('table');
    const tabs = document.querySelectorAll('.oc-tab');

    const toggles = {
        person: $('tg-person'),
        vehicle: $('tg-vehicle'),
        everyday: $('tg-everyday'),
        line: $('tg-line'),
        trails: $('tg-trails'),
    };

    const GROUP_ORDER = ['person', 'vehicle', 'everyday'];
    const GROUP_COLORS = { person: '#3cc850', vehicle: '#ffa500', everyday: '#288ce6' };

    let sourceMode = 'camera';  // camera | file
    let upload = null;          // { file_id, source, name }
    let active = false;         // frame + status loops running
    let frameSeq = -1;
    let currentUrl = null;
    let statusTimer = null;
    let lastState = 'idle';
    let lastSource = null;

    // ------------------------------------------------------------ config
    function readConfig() {
        return {
            groups: {
                person: toggles.person.checked,
                vehicle: toggles.vehicle.checked,
                everyday: toggles.everyday.checked,
            },
            line_enabled: toggles.line.checked,
            line_orientation: document.querySelector('input[name="line-orient"]:checked').value,
            line_position: Number(linePos.value) / 100,
            confidence: Number(conf.value) / 100,
            show_trails: toggles.trails.checked,
            model: modelSelect.value,
        };
    }

    function applyConfig(cfg) {
        if (!cfg) return;
        for (const g of GROUP_ORDER) toggles[g].checked = !!cfg.groups[g];
        toggles.line.checked = !!cfg.line_enabled;
        toggles.trails.checked = !!cfg.show_trails;
        const radio = document.querySelector(`input[name="line-orient"][value="${cfg.line_orientation}"]`);
        if (radio) radio.checked = true;
        linePos.value = Math.round(cfg.line_position * 100);
        conf.value = Math.round(cfg.confidence * 100);
        if (cfg.model) modelSelect.value = cfg.model;
        refreshLabels();
    }

    function refreshLabels() {
        linePosVal.textContent = `${linePos.value}%`;
        confVal.textContent = `${conf.value}%`;
        lineOpts.classList.toggle('disabled', !toggles.line.checked);
        const noLine = !toggles.line.checked || lastSource === 'image';
        groupCards.classList.toggle('no-line', noLine);
        labelTable.classList.toggle('no-line', noLine);
        for (const g of GROUP_ORDER) {
            groupCards.querySelector(`[data-group="${g}"]`).classList.toggle('off', !toggles[g].checked);
        }
        const horizontal = document.querySelector('input[name="line-orient"]:checked').value === 'horizontal';
        document.querySelectorAll('.dir-fwd').forEach((el) => { el.textContent = horizontal ? 'xuống' : 'sang phải'; });
        document.querySelectorAll('.dir-bwd').forEach((el) => { el.textContent = horizontal ? 'lên' : 'sang trái'; });
        stageHint.hidden = !(active && toggles.line.checked && lastSource !== 'image');
    }

    let configTimer = null;
    function pushConfig(delay = 0) {
        refreshLabels();
        clearTimeout(configTimer);
        configTimer = setTimeout(() => {
            fetch('/api/objects/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(readConfig()),
            }).catch(() => {});
        }, delay);
    }

    Object.values(toggles).forEach((el) => el.addEventListener('change', () => pushConfig()));
    document.querySelectorAll('input[name="line-orient"]').forEach((el) => el.addEventListener('change', () => pushConfig()));
    linePos.addEventListener('input', () => pushConfig(120));
    conf.addEventListener('input', () => pushConfig(120));
    modelSelect.addEventListener('change', () => pushConfig());

    // Click on the video to move the counting line there
    feed.addEventListener('click', (e) => {
        if (!toggles.line.checked || lastSource === 'image') return;
        const rect = feed.getBoundingClientRect();
        const horizontal = document.querySelector('input[name="line-orient"]:checked').value === 'horizontal';
        const ratio = horizontal ? (e.clientY - rect.top) / rect.height : (e.clientX - rect.left) / rect.width;
        linePos.value = Math.min(95, Math.max(5, Math.round(ratio * 100)));
        pushConfig();
    });

    // ------------------------------------------------------------ source
    tabs.forEach((tab) => tab.addEventListener('click', () => {
        if (active) return;
        tabs.forEach((t) => t.classList.toggle('active', t === tab));
        sourceMode = tab.dataset.source === 'file' ? 'file' : 'camera';
        fileRow.classList.toggle('hidden', sourceMode !== 'file');
        placeholderText.textContent = sourceMode === 'camera'
            ? 'Bấm "Bắt đầu" để mở camera và nhận diện'
            : (upload ? `Đã sẵn sàng: ${upload.name}. Bấm "Bắt đầu".` : 'Chọn một video hoặc ảnh để thử');
    }));

    $('choose-file-btn').addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', async () => {
        const file = fileInput.files[0];
        if (!file) return;
        fileName.textContent = `Đang tải lên: ${file.name}`;
        startBtn.disabled = true;
        const form = new FormData();
        form.append('file', file);
        try {
            const res = await fetch('/api/objects/upload', { method: 'POST', body: form });
            const data = await res.json();
            if (!data.success) throw new Error(data.error || 'Tải lên thất bại');
            upload = data;
            const kind = data.source === 'image' ? 'Ảnh' : 'Video';
            fileName.textContent = `${kind}: ${data.name}`;
            placeholderText.textContent = `Đã sẵn sàng: ${data.name}. Bấm "Bắt đầu".`;
        } catch (err) {
            upload = null;
            fileName.textContent = `Lỗi: ${err.message}`;
        } finally {
            startBtn.disabled = false;
            fileInput.value = '';
        }
    });

    // ------------------------------------------------------------ run control
    startBtn.addEventListener('click', async () => {
        if (sourceMode === 'file' && !upload) {
            fileInput.click();
            return;
        }
        startBtn.disabled = true;
        const body = { source: sourceMode === 'camera' ? 'camera' : 'file', config: readConfig() };
        if (upload && sourceMode === 'file') body.file_id = upload.file_id;
        try {
            const res = await fetch('/api/objects/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });
            const data = await res.json();
            if (!data.success) throw new Error(data.error || 'Không bắt đầu được');
            frameSeq = -1;
            setActive(true);
        } catch (err) {
            setState('error', err.message);
            startBtn.disabled = false;
        }
    });

    stopBtn.addEventListener('click', async () => {
        await fetch('/api/objects/stop', { method: 'POST' }).catch(() => {});
        await pollStatus();
        setActive(false);
    });

    resetBtn.addEventListener('click', () => {
        fetch('/api/objects/reset', { method: 'POST' }).then(pollStatus).catch(() => {});
    });

    function setActive(on) {
        active = on;
        startBtn.disabled = on;
        stopBtn.disabled = !on;
        tabs.forEach((t) => { t.disabled = on; });
        if (on) {
            frameLoop();
            clearInterval(statusTimer);
            statusTimer = setInterval(pollStatus, 500);
            pollStatus();
        } else {
            clearInterval(statusTimer);
        }
        refreshLabels();
    }

    // Release the camera when leaving the page
    const stopOnLeave = () => {
        if (active && navigator.sendBeacon) navigator.sendBeacon('/api/objects/stop');
    };
    window.addEventListener('pagehide', stopOnLeave);

    // ------------------------------------------------------------ frames
    const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

    async function frameLoop() {
        while (active) {
            try {
                const res = await fetch(`/api/objects/frame?after=${frameSeq}`, { cache: 'no-store' });
                if (res.status === 200) {
                    frameSeq = Number(res.headers.get('X-Frame-Seq')) || frameSeq + 1;
                    const blob = await res.blob();
                    const url = URL.createObjectURL(blob);
                    feed.src = url;
                    if (currentUrl) URL.revokeObjectURL(currentUrl);
                    currentUrl = url;
                    feed.hidden = false;
                    placeholder.hidden = true;
                    await sleep(15);
                } else {
                    await sleep(50);
                }
            } catch (err) {
                await sleep(400);
            }
        }
    }

    // ------------------------------------------------------------ status
    const STATE_TEXT = {
        idle: 'Chưa chạy',
        loading_model: 'Đang tải model',
        running: 'Đang nhận diện',
        finished: 'Video đã chạy xong',
        error: 'Lỗi',
    };

    function setState(state, error) {
        stateDot.className = 'oc-dot ' + ({ running: 'running', loading_model: 'loading', error: 'error', finished: 'finished' }[state] || '');
        stateText.textContent = error ? `${STATE_TEXT[state] || state}: ${error}` : (STATE_TEXT[state] || state);
    }

    async function pollStatus() {
        let s;
        try {
            const res = await fetch('/api/objects/status', { cache: 'no-store' });
            s = await res.json();
        } catch (err) {
            return;
        }
        lastState = s.state;
        lastSource = s.source;
        let stateLabel = s.state;
        if (s.state === 'loading_model' && !s.model_downloaded) {
            setState(stateLabel, null);
            stateText.textContent = 'Đang tải model lần đầu, chờ chút';
        } else {
            setState(stateLabel, s.error);
            if (s.state === 'running' && s.source === 'image') stateText.textContent = 'Ảnh tĩnh: chỉ đếm đối tượng trong khung';
        }

        const parts = [];
        if (s.source && s.source !== 'image' && s.fps) parts.push(`${s.fps} FPS`);
        if (s.inference_ms) parts.push(`suy luận ${s.inference_ms} ms`);
        if (s.frame_size) parts.push(`${s.frame_size[0]}x${s.frame_size[1]}`);
        if (s.model_name) parts.push(s.model_name);
        metrics.textContent = ['running', 'finished'].includes(s.state) ? parts.join('  |  ') : '';

        const showProgress = s.source === 'video' && s.progress !== null && s.progress !== undefined;
        progressWrap.classList.toggle('hidden', !showProgress);
        if (showProgress) {
            progressFill.style.width = `${s.progress}%`;
            progressText.textContent = `${Math.round(s.progress)}%`;
        }

        renderStats(s);

        if (active && ['idle', 'error', 'finished'].includes(s.state)) {
            // Fetch the last frame once more, then stop the loops
            setTimeout(() => setActive(false), 600);
        }
        refreshLabels();
    }

    function renderStats(s) {
        if (!s.groups) return;
        for (const g of GROUP_ORDER) {
            const data = s.groups[g];
            const card = groupCards.querySelector(`[data-group="${g}"]`);
            card.querySelectorAll('[data-k]').forEach((el) => { el.textContent = data[el.dataset.k]; });
        }

        const rows = [];
        for (const g of GROUP_ORDER) {
            if (!toggles[g].checked) continue;
            const labels = Object.entries(s.groups[g].labels || {})
                .filter(([, v]) => v.unique > 0 || v.current > 0)
                .sort((a, b) => b[1].current - a[1].current || b[1].unique - a[1].unique);
            for (const [, v] of labels) rows.push({ group: g, ...v });
        }

        labelRows.replaceChildren();
        if (!rows.length) {
            const tr = document.createElement('tr');
            tr.className = 'empty';
            const td = document.createElement('td');
            td.colSpan = 4;
            td.textContent = 'Chưa có dữ liệu';
            tr.appendChild(td);
            labelRows.appendChild(tr);
            return;
        }
        for (const r of rows) {
            const tr = document.createElement('tr');
            const name = document.createElement('td');
            const wrap = document.createElement('span');
            wrap.className = 'lbl';
            const sw = document.createElement('span');
            sw.className = `oc-swatch ${r.group}`;
            sw.style.background = GROUP_COLORS[r.group];
            wrap.append(sw, document.createTextNode(r.label_vi));
            name.appendChild(wrap);
            const cur = document.createElement('td');
            cur.textContent = r.current;
            const uniq = document.createElement('td');
            uniq.textContent = r.unique;
            const cross = document.createElement('td');
            cross.className = 'col-cross';
            cross.textContent = `${r.cross_fwd} / ${r.cross_bwd}`;
            tr.append(name, cur, uniq, cross);
            labelRows.appendChild(tr);
        }
    }

    // ------------------------------------------------------------ init
    applyConfig(window.OC_DEFAULT_CONFIG);
    fetch('/api/objects/status').then((r) => r.json()).then((s) => {
        if (s.config) applyConfig(s.config);
        if (s.state === 'running' || s.state === 'loading_model') {
            // A session is still running (e.g. page reload): attach to it
            setActive(true);
        } else {
            renderStats(s);
        }
    }).catch(() => {});
});
