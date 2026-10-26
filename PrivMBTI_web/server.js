const http = require('http');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { spawnSync } = require('child_process');

const PORT = Number(process.env.PORT || 3000);
const ROOT = __dirname;
const DATA_DIR = path.join(ROOT, 'data');

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.woff2': 'font/woff2',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.svg': 'image/svg+xml'
};

const PERSONA_TRAIT_TEMPLATE = {
  five_person_traits: {
    openness: null,
    conscientiousness: null,
    extraversion: null,
    agreeableness: null,
    neuroticism: null
  },
  attitudes_and_perceptions: {
    privacy_concern: null,
    data_concerns: null,
    online_trust: null,
    home_network_trust: null,
    threat_source: null,
    view_of_others_awareness: null,
    privacy_protection_actions: null,
    security_vs_convenience: null,
    social_vs_privacy: null,
    attitudes_toward_being_monitored: null,
    data_collecting_entity_trust: null
  },
  knowledge_and_learning: {
    knowledge_of_security_privacy_concepts: null,
    sources_of_security_knowledge: [],
    types_of_perceived_threats: []
  },
  behaviors_and_practices: {
    daily_online_hours: null,
    completeness_of_fallback_authentication: null,
    ability_to_help_others: null,
    behavior_changes: null,
    initiative_on_reviewing_policies: null,
    proactive_or_reactive: null,
    number_of_passwords: null,
    tendency_to_censor_information: null,
    social_networking_use: null,
    system_protection: [],
    tendency_to_adjust_website_settings: null,
    password_layer: null,
    recording_passwords: null
  }
};

function send(res, status, body, headers = {}) {
  res.writeHead(status, {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    ...headers
  });
  res.end(body);
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    let body = '';
    req.on('data', chunk => {
      body += chunk;
      if (body.length > 5 * 1024 * 1024) {
        reject(new Error('Request body too large'));
        req.destroy();
      }
    });
    req.on('end', () => resolve(body));
    req.on('error', reject);
  });
}

function csvCell(value) {
  const text = value == null ? '' : String(value);
  return `"${text.replace(/"/g, '""')}"`;
}

function getCsvHeader() {
  return [
    'savedAt',
    'sessionId',
    'startedAt',
    'completedAt',
    'totalMs',
    'resultCode',
    'personaName',
    'scoreV',
    'scoreC',
    'scoreU',
    'scoreL',
    'questionNumber',
    'questionId',
    'section',
    'trait',
    'type',
    'rawAnswer',
    'answerValue',
    'answeredAt',
    'elapsedMs'
  ];
}

function timestampForFileName(date) {
  const pad = (value, size = 2) => String(value).padStart(size, '0');
  return [
    date.getFullYear(),
    pad(date.getMonth() + 1),
    pad(date.getDate())
  ].join('') + '-' + [
    pad(date.getHours()),
    pad(date.getMinutes()),
    pad(date.getSeconds())
  ].join('') + '-' + pad(date.getMilliseconds(), 3);
}

function safeFilePart(value) {
  // Only replace characters that are unsafe for filesystems:
  // Windows forbidden: < > : " / \ | ? *
  // Also strip control characters and leading/trailing dots/spaces
  return String(value || 'unknown')
    .replace(/[<>:"/\\|?*\x00-\x1f]/g, '-')
    .replace(/^\.+|\.+$/g, '')
    .trim()
    .slice(0, 80);
}

function buildSessionDirName(payload, fileStamp, resultCode) {
  const participant = payload.participant || {};
  const number = participant.number == null ? 'unknown-number' : `no-${participant.number}`;
  const name = safeFilePart(participant.name || 'unknown-name');
  if (participant.number != null || participant.name) {
    return `participant-${safeFilePart(number)}-${name}`;
  }
  return `privacy-mbti-${fileStamp}-${safeFilePart(resultCode)}`;
}

function buildPersonaId(completedAt, sessionId) {
  return crypto
    .createHash('sha256')
    .update(`${completedAt || ''}|${sessionId || ''}`)
    .digest('hex');
}

function scorePersonaWithPython(payload) {
  const scriptPath = path.join(ROOT, '4-dimensional_personality.py');
  const pythonCandidates = [process.env.PYTHON || 'python', 'py'];

  for (const pythonCommand of pythonCandidates) {
    const result = spawnSync(pythonCommand, [scriptPath, '--stdin'], {
      input: JSON.stringify(payload),
      encoding: 'utf8',
      maxBuffer: 1024 * 1024
    });

    if (result.status === 0 && result.stdout) {
      try {
        return JSON.parse(result.stdout);
      } catch (err) {
        // fall through to the next candidate or payload fallback
      }
    }
  }

  return null;
}

function createEmptyPersonaTraits() {
  return JSON.parse(JSON.stringify(PERSONA_TRAIT_TEMPLATE));
}

function mergePersonaTraits(base, updates) {
  const merged = createEmptyPersonaTraits();
  for (const [section, traits] of Object.entries(base || {})) {
    if (!merged[section]) merged[section] = {};
    for (const [trait, value] of Object.entries(traits || {})) {
      merged[section][trait] = Array.isArray(value) ? [...value] : value;
    }
  }
  for (const [section, traits] of Object.entries(updates || {})) {
    if (!merged[section]) merged[section] = {};
    for (const [trait, value] of Object.entries(traits || {})) {
      merged[section][trait] = Array.isArray(value) ? [...value] : value;
    }
  }
  return merged;
}

function answerValue(answer) {
  if (answer.type === 'likert5') {
    return [
      'strongly_disagree',
      'disagree',
      'neutral',
      'agree',
      'strongly_agree'
    ][answer.rawAnswer] || String((answer.rawAnswer != null ? answer.rawAnswer : ''));
  }
  if (Array.isArray(answer.rawAnswer)) return answer.rawAnswer.join(';');
  return String((answer.rawAnswer != null ? answer.rawAnswer : ''));
}

function buildAnswersRecord(answersData) {
  // Handle new object format: { startedAt, completedAt, totalMs, items: [...] }
  if (answersData && !Array.isArray(answersData) && Array.isArray(answersData.items)) {
    return {
      startedAt: answersData.startedAt,
      completedAt: answersData.completedAt,
      totalMs: answersData.totalMs,
      items: answersData.items.map(answer => ({
        questionNumber: answer.questionNumber,
        questionId: answer.questionId,
        section: answer.section,
        trait: answer.trait,
        type: answer.type,
        rawAnswer: answer.rawAnswer,
        answerValue: answerValue(answer),
        answeredAt: answer.answeredAt,
        elapsedMs: answer.elapsedMs
      }))
    };
  }
  // Handle legacy array format
  if (Array.isArray(answersData)) {
    return {
      startedAt: null,
      completedAt: null,
      totalMs: null,
      items: answersData.map(answer => ({
        questionNumber: answer.questionNumber,
        questionId: answer.questionId,
        section: answer.section,
        trait: answer.trait,
        type: answer.type,
        rawAnswer: answer.rawAnswer,
        answerValue: answerValue(answer),
        answeredAt: answer.answeredAt,
        elapsedMs: answer.elapsedMs
      }))
    };
  }
  // Fallback
  return { startedAt: null, completedAt: null, totalMs: null, items: [] };
}

function cleanRecord(payload, savedAt) {
  return {
    savedAt,
    sessionId: payload.sessionId,
    participant: payload.participant || null,
    startedAt: payload.startedAt,
    completedAt: payload.completedAt,
    totalMs: payload.totalMs,
    result: {
      code: (payload.result != null ? payload.result.code : void 0),
      personaName: (payload.result != null ? payload.result.personaName : void 0),
      scores: (payload.result != null ? payload.result.scores : void 0)
    },
    personaTraits: payload.personaTraits || null,
    questionnaire: payload.questionnaire || null,
    feedback: payload.feedback || null,
    attentionChecks: payload.attentionChecks || null,
    answers: buildAnswersRecord(payload.answers)
  };
}

function toBeijingISO(ms) {
  if (!ms) return null;
  const bj = new Date(ms + 8 * 3600 * 1000);
  return bj.toISOString().replace('Z', '+08:00');
}

function appendResult(payload) {
  fs.mkdirSync(DATA_DIR, { recursive: true });
  const savedAtDate = new Date();
  const savedAt = toBeijingISO(savedAtDate.getTime());
  const fileStamp = timestampForFileName(savedAtDate);
  const pythonResult = scorePersonaWithPython(payload);
  const resolvedResult = pythonResult || payload.result || {};
  const record = cleanRecord({
    ...payload,
    result: {
      code: resolvedResult.persona_code || resolvedResult.code,
      personaName: resolvedResult.personaName || resolvedResult.persona_name,
      scores: resolvedResult.scores
    }
  }, savedAt);
  const sessionDir = path.join(DATA_DIR, buildSessionDirName(payload, fileStamp, (record.result != null ? record.result.code : void 0)));
  const jsonPath = path.join(sessionDir, 'result.json');
  const personaPath = path.join(sessionDir, 'persona.json');
  const csvPath = path.join(sessionDir, 'answers.csv');
  const questionnaireJsonPath = path.join(sessionDir, 'questionnaire.json');
  const questionnaireCsvPath = path.join(sessionDir, 'questionnaire_answers.csv');

  fs.mkdirSync(sessionDir, { recursive: true });

  fs.writeFileSync(jsonPath, `${JSON.stringify(record, null, 2)}\n`, 'utf8');
  const personaCode = resolvedResult.persona_code || resolvedResult.code || 'unknown';
  const personaRecord = {
    persona_traits: mergePersonaTraits(resolvedResult.persona_traits || payload.personaTraits, {}),
    persona_code: personaCode,
    persona_id: buildPersonaId(payload.completedAt || savedAt, payload.sessionId)
  };
  fs.writeFileSync(personaPath, `${JSON.stringify(personaRecord, null, 2)}\n`, 'utf8');

  const rows = (record.answers && Array.isArray(record.answers.items)) ? record.answers.items : [];
  const csvRows = rows.map(answer => [
    record.savedAt,
    record.sessionId,
    record.startedAt,
    record.completedAt,
    record.totalMs,
    (record.result != null ? record.result.code : void 0),
    (record.result != null ? record.result.personaName : void 0),
    (record.result != null && record.result.scores != null ? record.result.scores.V : void 0),
    (record.result != null && record.result.scores != null ? record.result.scores.C : void 0),
    (record.result != null && record.result.scores != null ? record.result.scores.U : void 0),
    (record.result != null && record.result.scores != null ? record.result.scores.L : void 0),
    answer.questionNumber,
    answer.questionId,
    answer.section,
    answer.trait,
    answer.type,
    JSON.stringify(answer.rawAnswer),
    answer.answerValue,
    answer.answeredAt,
    answer.elapsedMs
  ].map(csvCell).join(',')).join('\n');

  const csv = [getCsvHeader().join(','), csvRows].filter(Boolean).join('\n');
  fs.writeFileSync(csvPath, `\ufeff${csv}\n`, 'utf8');

  if (record.questionnaire) {
    fs.writeFileSync(questionnaireJsonPath, `${JSON.stringify(record.questionnaire, null, 2)}\n`, 'utf8');
    const questionnaireRows = Array.isArray(record.questionnaire.answers)
      ? record.questionnaire.answers.map(answer => [
        record.savedAt,
        record.sessionId,
        (record.participant != null ? record.participant.number : void 0),
        (record.participant != null ? record.participant.name : void 0),
        record.questionnaire.number,
        record.questionnaire.url,
        answer.itemNumber,
        answer.itemId,
        answer.source,
        answer.type,
        answer.taskNumber,
        answer.questionNumber,
        answer.construct,
        answer.appName,
        answer.question,
        JSON.stringify(answer.answer),
        answer.answeredAt,
        answer.elapsedMs
      ].map(csvCell).join(',')).join('\n')
      : '';
    const questionnaireHeader = [
      'savedAt',
      'sessionId',
      'participantNumber',
      'participantName',
      'questionnaireNumber',
      'questionnaireUrl',
      'itemNumber',
      'itemId',
      'source',
      'type',
      'taskNumber',
      'questionNumber',
      'construct',
      'appName',
      'question',
      'answer',
      'answeredAt',
      'elapsedMs'
    ].join(',');
    const questionnaireCsv = [questionnaireHeader, questionnaireRows].filter(Boolean).join('\n');
    fs.writeFileSync(questionnaireCsvPath, `\ufeff${questionnaireCsv}\n`, 'utf8');
  }

  // \u2500\u2500 Save attention check results \u2500\u2500
  if (Array.isArray(record.attentionChecks) && record.attentionChecks.length) {
    const attentionCheckJsonPath = path.join(sessionDir, 'attention_check.json');
    const attentionCheckCsvPath = path.join(sessionDir, 'attention_check.csv');
    fs.writeFileSync(attentionCheckJsonPath, `${JSON.stringify(record.attentionChecks, null, 2)}\n`, 'utf8');

    const attentionCheckHeader = [
      'savedAt',
      'sessionId',
      'participantNumber',
      'participantName',
      'phase',
      'answer',
      'passed',
      'answeredAt',
      'elapsedMs'
    ].join(',');
    const attentionCheckRows = record.attentionChecks.map(ac => [
      record.savedAt,
      record.sessionId,
      (record.participant != null ? record.participant.number : void 0),
      (record.participant != null ? record.participant.name : void 0),
      ac.phase,
      ac.answer,
      ac.passed,
      ac.answeredAt,
      ac.elapsedMs
    ].map(csvCell).join(',')).join('\n');
    const attentionCheckCsv = [attentionCheckHeader, attentionCheckRows].filter(Boolean).join('\n');
    fs.writeFileSync(attentionCheckCsvPath, `\ufeff${attentionCheckCsv}\n`, 'utf8');
  }

  // \u2500\u2500 Save feedback results \u2500\u2500
  if (Array.isArray(record.feedback) && record.feedback.length) {
    const feedbackJsonPath = path.join(sessionDir, 'feedback.json');
    const feedbackCsvPath = path.join(sessionDir, 'feedback.csv');
    fs.writeFileSync(feedbackJsonPath, `${JSON.stringify(record.feedback, null, 2)}\n`, 'utf8');

    const feedbackHeader = [
      'savedAt',
      'sessionId',
      'participantNumber',
      'participantName',
      'questionId',
      'question',
      'choice',
      'choiceIndex',
      'followUpQuestion',
      'followUpText'
    ].join(',');
    const feedbackRows = record.feedback.map((fb, index) => [
      record.savedAt,
      record.sessionId,
      (record.participant != null ? record.participant.number : void 0),
      (record.participant != null ? record.participant.name : void 0),
      `feedback-${index + 1}`,
      fb.question,
      fb.choice,
      fb.choiceIndex,
      fb.followUpQuestion,
      fb.followUpText
    ].map(csvCell).join(',')).join('\n');
    const feedbackCsv = [feedbackHeader, feedbackRows].filter(Boolean).join('\n');
    fs.writeFileSync(feedbackCsvPath, `\ufeff${feedbackCsv}\n`, 'utf8');
  }

  return {
    dir: sessionDir,
    json: jsonPath,
    persona: personaPath,
    csv: csvPath,
    questionnaireJson: record.questionnaire ? questionnaireJsonPath : null,
    questionnaireCsv: record.questionnaire ? questionnaireCsvPath : null,
    result: resolvedResult || null
  };
}

function serveFile(req, res) {
  const url = new URL(req.url, `http://${req.headers.host}`);
  const pathname = decodeURIComponent(url.pathname);
  const relativePath = pathname === '/' ? 'index.html' : pathname.slice(1);
  const filePath = path.resolve(ROOT, relativePath);

  if (!filePath.startsWith(ROOT)) {
    send(res, 403, 'Forbidden', { 'Content-Type': 'text/plain; charset=utf-8' });
    return;
  }

  fs.readFile(filePath, (err, data) => {
    if (err) {
      send(res, 404, 'Not found', { 'Content-Type': 'text/plain; charset=utf-8' });
      return;
    }
    const headers = {
      'Content-Type': MIME[path.extname(filePath).toLowerCase()] || 'application/octet-stream',
      'Cache-Control': 'no-cache, no-store, must-revalidate'
    };
    send(res, 200, data, headers);
  });
}

const server = http.createServer(async (req, res) => {
  if (req.method === 'OPTIONS') {
    send(res, 204, '');
    return;
  }

  if (req.method === 'POST' && req.url === '/api/quiz-results') {
    try {
      const body = await readBody(req);
      const payload = JSON.parse(body);
      const hasAnswers = payload.answers && (Array.isArray(payload.answers) || Array.isArray(payload.answers.items));
      if (!payload || !payload.sessionId || !hasAnswers) {
        send(res, 400, JSON.stringify({ ok: false, error: 'Invalid payload' }), {
          'Content-Type': 'application/json; charset=utf-8'
        });
        return;
      }
      const savedTo = appendResult(payload);
      send(res, 200, JSON.stringify({ ok: true, savedTo }), {
        'Content-Type': 'application/json; charset=utf-8'
      });
    } catch (err) {
      send(res, 500, JSON.stringify({ ok: false, error: err.message }), {
        'Content-Type': 'application/json; charset=utf-8'
      });
    }
    return;
  }

  if (req.method === 'GET') {
    serveFile(req, res);
    return;
  }

  send(res, 405, 'Method not allowed', { 'Content-Type': 'text/plain; charset=utf-8' });
});

server.listen(PORT, () => {
  console.log(`Privacy MBTI local server running at http://localhost:${PORT}`);
  console.log(`Results will be saved to ${DATA_DIR}`);
});
