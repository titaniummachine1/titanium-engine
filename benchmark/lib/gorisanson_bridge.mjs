/**
 * Bridge between scraped UI actions and gorisanson Game move tuples.
 * Gorisanson row 0 = top; UI row 1 = bottom.
 */

const PAWN_ROWS = 9;
const WALL_ROWS = 8;

export function actionToGorisansonMove(action) {
  const col = action.coordinate.column.charCodeAt(0) - 97;
  if (action.wallType === 'h') {
    const row = WALL_ROWS - action.coordinate.row;
    return [null, [row, col], null];
  }
  if (action.wallType === 'v') {
    const row = WALL_ROWS - action.coordinate.row;
    return [null, null, [row, col]];
  }
  const row = PAWN_ROWS - action.coordinate.row;
  return [[row, col], null, null];
}

export function gorisansonMoveToAction(move) {
  const [pawn, horiz, vert] = move;
  if (pawn) {
    const [row, col] = pawn;
    return {
      coordinate: { column: String.fromCharCode(97 + col), row: PAWN_ROWS - row },
    };
  }
  if (horiz) {
    const [row, col] = horiz;
    return {
      coordinate: { column: String.fromCharCode(97 + col), row: WALL_ROWS - row },
      wallType: 'h',
    };
  }
  const [row, col] = vert;
  return {
    coordinate: { column: String.fromCharCode(97 + col), row: WALL_ROWS - row },
    wallType: 'v',
  };
}
