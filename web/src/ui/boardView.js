import { WallType, formatCoordinate, toAlgebraic } from '../lib/gameLogic.js';
import { playerColorName } from '../lib/playerColors.js';
import {
  catSquareOverlay,
  catWallOutlineColor,
  catSquareIndex,
  isSquareSkipped,
} from '../lib/catHeatmap.js';
import {
  lmrDepthStyle,
  lmrDisplayText,
  lmrSubLabel,
  lmrWallOutlineColor,
} from '../lib/lmrHeatmap.js';
import './board.css';

const SQUARE_TRACK = '9fr';
const WALL_TRACK = '2fr';

function indexToColumnLocal(index) {
  return String.fromCharCode(index + 96);
}

function buildGridTracks(count) {
  return Array.from({ length: count }, (_, index) => (index % 2 === 0 ? SQUARE_TRACK : WALL_TRACK)).join(' ');
}

function columnLabel(colIndex) {
  return indexToColumnLocal(colIndex);
}

function rowLabel(rowIndex, numRows) {
  return String(numRows - rowIndex);
}

export function renderBoard(container, state, controller) {
  const {
    board,
    validActions,
    playerPositions,
    wallsRemaining,
    winner,
    playerToMove,
    settings,
    engineStatus,
    engineErrors,
    aiThinking,
    uiMode,
    catViz,
    lmrViz,
  } = state;

  const numRows = board.numRows();
  const numCols = board.numColumns();
  const validKeys = new Set(validActions.map((action) => toAlgebraic(action)));

  const wallOwners = new Map();
  for (const [playerNum, coordinate, wallType] of state.wallsByPlayer) {
    wallOwners.set(toAlgebraic({ coordinate, wallType }), playerNum);
  }

  const lastKey = state.lastAction ? toAlgebraic(state.lastAction) : null;
  const freePlay = uiMode === 'analysis';
  const canInteract =
    !winner &&
    !aiThinking &&
    uiMode !== 'replay' &&
    (freePlay || controller.session.isHumanTurn(settings.players));
  const showCat = settings.showCatVision && catViz && uiMode !== 'replay';
  const showLmr = settings.showLmrVision && lmrViz && uiMode !== 'replay';
  const showCoords = settings.displayCoordinates;
  const showWallCounts = settings.displayRemainingWalls;
  const isRotated = settings.rotateBoard;

  container.innerHTML = '';
  container.className = 'board-panel';

  const boardShell = document.createElement('div');
  boardShell.className =
    'board' +
    (isRotated ? ' board--rotate' : '') +
    (showCat ? ' board--cat' : '') +
    (showLmr ? ' board--lmr' : '') +
    (freePlay ? ' board--freeplay' : '');

  const engineStateP1 = document.createElement('div');
  engineStateP1.className = 'engine-state engine-state--p1';
  engineStateP1.appendChild(
    renderTurnIndicator(1, playerToMove, settings.players[0], engineStatus, engineErrors, aiThinking, winner, freePlay),
  );

  const engineStateP2 = document.createElement('div');
  engineStateP2.className = 'engine-state engine-state--p2';
  engineStateP2.appendChild(
    renderTurnIndicator(2, playerToMove, settings.players[1], engineStatus, engineErrors, aiThinking, winner, freePlay),
  );

  const coordLabelsRow = renderCoordinateLabels('row', numRows, showCoords, controller);
  const coordLabelsCol = renderCoordinateLabels('col', numCols, showCoords, controller);

  const wallMarksP1 = renderWallMarks(1, wallsRemaining[0], showWallCounts, controller);
  const wallMarksP2 = renderWallMarks(2, wallsRemaining[1], showWallCounts, controller);

  const grid = document.createElement('div');
  grid.className = 'board-grid';
  grid.style.gridTemplateColumns = buildGridTracks(numCols * 2 - 1);
  grid.style.gridTemplateRows = buildGridTracks(numRows * 2 - 1);

  for (let p = 0; p < numRows * 2 - 1; p++) {
    for (let h = 0; h < numCols * 2 - 1; h++) {
      grid.appendChild(
        renderBoardCell({
          p,
          h,
          numRows,
          numCols,
          playerPositions,
          validKeys,
          wallOwners,
          lastKey,
          canInteract,
          playerToMove,
          catViz: showCat ? catViz : null,
          lmrViz: showLmr ? lmrViz : null,
        }),
      );
    }
  }

  boardShell.append(
    engineStateP1,
    engineStateP2,
    wallMarksP1,
    wallMarksP2,
    coordLabelsRow,
    coordLabelsCol,
    grid,
  );
  container.appendChild(boardShell);

  if (winner) {
    const banner = document.createElement('div');
    banner.className = 'winner-banner';
    banner.textContent = `${playerColorName(winner)} wins!`;
    container.appendChild(banner);
  }

  boardShell.addEventListener('click', (event) => {
    const actionNode = event.target.closest?.('[data-action]');
    if (!actionNode) {
      return;
    }
    if (actionNode.dataset.isValid !== 'true') {
      return;
    }

    const actionKey = actionNode.dataset.action;
    if (!actionKey) {
      return;
    }

    if (actionKey.length === 2) {
      controller.tryAction({ coordinate: parseCoord(actionKey) });
      return;
    }

    const wallType = actionKey[2] === 'h' ? WallType.Horizontal : WallType.Vertical;
    controller.tryAction({
      coordinate: parseCoord(actionKey.slice(0, 2)),
      wallType,
    });
  });
}

function renderBoardCell({
  p,
  h,
  numRows,
  numCols,
  playerPositions,
  validKeys,
  wallOwners,
  lastKey,
  canInteract,
  playerToMove,
  catViz,
  lmrViz,
}) {
  const row = numRows - Math.floor(p / 2);
  const col = Math.floor(h / 2) + 1;
  const isEvenRow = p % 2 === 0;
  const isEvenCol = h % 2 === 0;

  let cellType;
  if (isEvenRow && isEvenCol) {
    cellType = 'square';
  } else if (isEvenRow) {
    cellType = 'verticalWall';
  } else if (isEvenCol) {
    cellType = 'horizontalWall';
  } else {
    cellType = 'wallIntersection';
  }

  const cell = document.createElement('div');
  cell.dataset.cellType = cellType;
  cell.dataset.coordinate = formatCoordinate({ row, column: indexToColumnLocal(col) });

  if (cellType === 'square') {
    const coordinate = { row, column: indexToColumnLocal(col) };
    const key = formatCoordinate(coordinate);
    const pawnPlayer = playerPositions.findIndex(
      (pos) => pos.row === coordinate.row && pos.column === coordinate.column,
    );
    const isValid = validKeys.has(key) && canInteract && pawnPlayer < 0;
    const isPrev = lastKey === key;
    const engineRow = row - 1;
    const engineCol = col - 1;
    const sqIdx = catSquareIndex(engineRow, engineCol);
    const heat = catViz?.squares?.[sqIdx] ?? 0;
    const reachable =
      catViz?.reachable == null ? true : catViz.reachable[sqIdx] === true;
    const skipped = catViz && isSquareSkipped(reachable);
    const overlay = catViz
      ? catSquareOverlay(heat, reachable, {
        coldCm: catViz.coldCm,
        hotCm: catViz.hotCm,
        maxCm: catViz.maxCm,
      })
      : null;

    const square = document.createElement('div');
    square.className = 'board-cell__square';
    square.classList.toggle('board-cell__square--prev', isPrev);
    square.classList.toggle('board-cell__square--valid', isValid);
    const lmrEntry = lmrViz?.moveIndex?.get(key);
    if (lmrViz && lmrEntry?.kind === 'pawn') {
      square.classList.add('board-cell__square--lmr');
      const style = lmrDepthStyle(lmrEntry, lmrViz);
      const tint = document.createElement('div');
      tint.className = 'board-cell__lmr-tint';
      tint.style.backgroundColor = style.fill;
      square.appendChild(tint);
      const val = document.createElement('span');
      val.className = 'board-cell__lmr-val';
      val.textContent = lmrDisplayText(lmrEntry, lmrViz);
      square.appendChild(val);
      const sub = lmrSubLabel(lmrEntry, lmrViz);
      if (sub) {
        const subEl = document.createElement('span');
        subEl.className = 'board-cell__lmr-sub';
        subEl.textContent = sub;
        square.appendChild(subEl);
      }
      const mode = lmrViz.shallow ? 'pre-search plan' : 'searched';
      square.title = `LMR ${mode}: ${style.label} · #${lmrEntry.order + 1}${lmrEntry.reSearched ? ' · re-search' : ''}`;
    }

    if (catViz) {
      square.classList.add('board-cell__square--cat');
      if (skipped) {
        square.classList.add('board-cell__square--skipped');
      }
      if (overlay) {
        const tint = document.createElement('div');
        tint.className = 'board-cell__cat-tint';
        tint.style.backgroundColor = overlay.fill;
        square.appendChild(tint);
      }
      // Raw engine heat in centi-squares — exactly what search sees, no scaling.
      if (!skipped && heat > 0) {
        const cold = catViz.coldCm ?? 60;
        const hot = catViz.hotCm ?? 160;
        const val = document.createElement('span');
        val.className =
          'board-cell__cat-val ' +
          (heat >= hot
            ? 'board-cell__cat-val--hot'
            : heat >= cold
              ? 'board-cell__cat-val--warm'
              : 'board-cell__cat-val--cold');
        val.textContent = String(heat);
        square.appendChild(val);
      }
    }
    square.dataset.action = key;
    square.dataset.isValid = String(isValid);
    if (catViz) {
      const cold = catViz.coldCm ?? 60;
      const hot = catViz.hotCm ?? 160;
      square.title = skipped
        ? 'Skipped — unreachable void'
        : heat >= hot
          ? `CAT hot ${heat} cm (tactical / no LMR)`
          : heat >= cold
            ? `CAT warm ${heat} cm (corridor)`
            : heat > 0
              ? `CAT cold ${heat} cm (LMR fringe)`
              : 'Off corridor — cold';
    }

    if (pawnPlayer >= 0) {
      const pawn = document.createElement('div');
      pawn.className = `board-cell__pawn board-cell__pawn--player${pawnPlayer + 1}`;
      square.appendChild(pawn);
    }

    cell.appendChild(square);
    return cell;
  }

  if (cellType === 'horizontalWall' || cellType === 'verticalWall') {
    const coordinate = {
      row: row - 1,
      column: indexToColumnLocal(col),
    };
    const wallType = cellType === 'horizontalWall' ? WallType.Horizontal : WallType.Vertical;
    const key = toAlgebraic({ coordinate, wallType });
    const owner = wallOwners.get(key);
    const isValid = validKeys.has(key) && canInteract;
    const isPrev = lastKey === key;
    const wallCat = catViz?.wallIndex?.get(key);

    const wall = document.createElement('div');
    wall.className = 'board-cell__wall';
    wall.classList.add(cellType === 'horizontalWall' ? 'board-cell__wall--h' : 'board-cell__wall--v');
    wall.dataset.action = key;
    wall.dataset.isValid = String(isValid);

    if (owner) {
      wall.classList.add('board-cell__wall--placed', `board-cell__wall--player${owner}`);
    } else if (isValid) {
      wall.classList.add('board-cell__wall--valid', `board-cell__wall--player${playerToMove}`);
      cell.classList.add('board-cell--wall-valid');
    }

    if (isPrev) {
      wall.style.zIndex = '1900';
    }

    cell.appendChild(wall);

    const lmrWall = lmrViz?.moveIndex?.get(key);
    if (lmrViz && lmrWall?.kind === 'wall' && !owner) {
      const hint = document.createElement('div');
      hint.className = 'board-cell__lmr-wall-hint';
      const style = lmrDepthStyle(lmrWall, lmrViz);
      hint.style.setProperty('--lmr-wall-color', lmrWallOutlineColor(lmrWall, lmrViz));
      hint.style.backgroundColor = style.fill;
      const tag = document.createElement('span');
      tag.className = 'board-cell__lmr-wall-tag';
      tag.textContent = lmrDisplayText(lmrWall, lmrViz);
      hint.appendChild(tag);
      const sub = lmrSubLabel(lmrWall, lmrViz);
      if (sub) {
        const subTag = document.createElement('span');
        subTag.className = 'board-cell__lmr-wall-sub';
        subTag.textContent = sub;
        hint.appendChild(subTag);
      }
      hint.title = `LMR ${style.label} · order ${lmrWall.order}`;
      cell.appendChild(hint);
    } else if (catViz && wallCat && !owner) {
      const hint = document.createElement('div');
      hint.className = 'board-cell__cat-wall-hint';
      if (wallCat.skip) {
        hint.classList.add('board-cell__cat-wall-hint--skipped');
      }
      const scale = {
        coldCm: catViz.coldCm,
        hotCm: catViz.hotCm,
        maxCm: catViz.maxCm,
      };
      const outline = catWallOutlineColor(wallCat.heat, scale);
      const fill = catSquareOverlay(wallCat.heat, true, scale);
      hint.style.setProperty('--cat-wall-color', outline);
      if (fill) {
        hint.style.setProperty('--cat-wall-fill', fill.fill);
      }
      if (wallCat.heat > 0) {
        const tag = document.createElement('span');
        tag.className = 'board-cell__cat-wall-tag';
        tag.textContent = String(wallCat.heat);
        hint.appendChild(tag);
      }
      hint.title = wallCat.skip
        ? `CAT skipped (pruned)`
        : wallCat.search
          ? `CAT ${wallCat.heat} cm (searchable)`
          : `CAT ${wallCat.heat} cm (cold fringe)`;
      cell.appendChild(hint);
    }
  }

  return cell;
}

function renderCoordinateLabels(axis, count, visible, controller) {
  const wrap = document.createElement('div');
  wrap.className =
    'coord-labels coord-labels--' +
    (axis === 'row' ? 'row' : 'col') +
    (visible ? ' coord-labels--visible' : '');
  wrap.addEventListener('click', () => controller.toggleDisplayCoordinates?.());

  for (let index = 0; index < count; index++) {
    if (index > 0) {
      const spacer = document.createElement('div');
      spacer.className = 'coord-labels__spacer';
      wrap.appendChild(spacer);
    }

    const label = document.createElement('span');
    label.className = 'coord-labels__label';
    label.textContent = axis === 'row' ? rowLabel(index, count) : columnLabel(index + 1);
    wrap.appendChild(label);
  }

  return wrap;
}

function renderWallMarks(playerNum, remaining, visible, controller) {
  const wrap = document.createElement('div');
  wrap.className = `wall-marks wall-marks--p${playerNum}`;
  wrap.addEventListener('click', () => controller.toggleDisplayRemainingWalls?.());

  const count = document.createElement('span');
  count.className = 'wall-marks__count' + (visible ? ' wall-marks__count--visible' : '');
  count.textContent = String(remaining);

  const slots = [];
  for (let index = 0; index < 10; index++) {
    const slot = document.createElement('div');
    const isAvailable = playerNum === 1 ? index < remaining : 10 - index <= remaining;
    slot.className =
      'wall-marks__slot' +
      (isAvailable ? ' wall-marks__slot--available' : '') +
      ` wall-marks__slot--player${playerNum}`;
    slots.push(slot);
  }

  if (playerNum === 2) {
    wrap.append(count, ...slots);
  } else {
    wrap.append(...slots, count);
  }

  return wrap;
}

function parseCoord(text) {
  return { column: text[0], row: Number.parseInt(text[1], 10) };
}

function renderTurnIndicator(playerNum, playerToMove, playerType, engineStatus, engineErrors, aiThinking, winner, freePlay) {
  const wrap = document.createElement('div');
  wrap.className = 'turn-indicator';

  if (winner || playerToMove !== playerNum) {
    return wrap;
  }

  if (freePlay || playerType === 'human') {
    const dot = document.createElement('div');
    dot.className = `turn-dot turn-dot--player${playerNum}`;
    dot.title = 'Your turn';
    wrap.appendChild(dot);
    return wrap;
  }

  const status = engineStatus[playerType] ?? 'idle';
  const spinner = document.createElement('div');
  spinner.className = 'engine-spinner';
  if (status === 'error') {
    spinner.classList.add('engine-spinner--error');
    spinner.textContent = '!';
    spinner.title = engineErrors?.[playerType]
      ? `Engine error: ${engineErrors[playerType]}`
      : 'Engine error — try New game or pick another opponent';
  } else if (status === 'pondering') {
    spinner.title = 'Pondering on opponent time...';
  } else if (aiThinking || status === 'searching') {
    spinner.title = 'Engine is thinking...';
  } else if (status === 'connecting') {
    spinner.title = 'Connecting to engine...';
  } else {
    spinner.classList.add('engine-spinner--error');
    spinner.textContent = '!';
    spinner.title = 'Engine idle on AI turn — try New game';
  }
  wrap.appendChild(spinner);
  return wrap;
}
