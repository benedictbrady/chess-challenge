use eframe::egui;
use engine::bot::{Bot, BaselineBot};
use engine::game::{GameState, Outcome};
use engine::{Color, File, Move, NnBot, Piece, Rank, Square};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

#[derive(Clone)]
struct SharedState {
    game: GameState,
    bot_thinking: bool,
    status_message: String,
}

impl SharedState {
    fn new() -> Self {
        SharedState {
            game: GameState::new(),
            bot_thinking: false,
            status_message: "White to move".to_string(),
        }
    }
}

struct ChessApp {
    shared: Arc<Mutex<SharedState>>,
    move_sender: std::sync::mpsc::Sender<Move>,
    selected_square: Option<Square>,
    legal_move_targets: Vec<Square>,
    human_color: Color,
    bot_vs_bot: bool,
}

impl ChessApp {
    fn new(
        shared: Arc<Mutex<SharedState>>,
        move_sender: std::sync::mpsc::Sender<Move>,
        bot_vs_bot: bool,
    ) -> Self {
        ChessApp {
            shared,
            move_sender,
            selected_square: None,
            legal_move_targets: Vec::new(),
            human_color: Color::White,
            bot_vs_bot,
        }
    }

    fn update_legal_targets(&mut self, from: Square) {
        let state = self.shared.lock().unwrap();
        self.legal_move_targets = state
            .game
            .legal_moves()
            .into_iter()
            .filter(|mv| mv.from == from)
            .map(|mv| mv.to)
            .collect();
    }

    fn piece_char(piece: Piece, color: Color) -> &'static str {
        match (piece, color) {
            (Piece::King, Color::White) => "♔",
            (Piece::Queen, Color::White) => "♕",
            (Piece::Rook, Color::White) => "♖",
            (Piece::Bishop, Color::White) => "♗",
            (Piece::Knight, Color::White) => "♘",
            (Piece::Pawn, Color::White) => "♙",
            (Piece::King, Color::Black) => "♚",
            (Piece::Queen, Color::Black) => "♛",
            (Piece::Rook, Color::Black) => "♜",
            (Piece::Bishop, Color::Black) => "♝",
            (Piece::Knight, Color::Black) => "♞",
            (Piece::Pawn, Color::Black) => "♟",
        }
    }
}

impl eframe::App for ChessApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint_after(std::time::Duration::from_millis(100));

        let (game_snapshot, bot_thinking, status_message) = {
            let state = self.shared.lock().unwrap();
            (state.game.clone(), state.bot_thinking, state.status_message.clone())
        };

        egui::SidePanel::right("info_panel").min_width(200.0).show(ctx, |ui| {
            if self.bot_vs_bot {
                ui.heading("NnBot vs BaselineBot");
                ui.label("NnBot (White) · BaselineBot (Black)");
            } else {
                ui.heading("Chess Challenge");
            }
            ui.separator();
            ui.label(&status_message);
            if bot_thinking {
                ui.label("Thinking...");
            }
            ui.separator();
            ui.heading("Move History");
            egui::ScrollArea::vertical().show(ui, |ui| {
                for (i, mv) in game_snapshot.history.iter().enumerate() {
                    let promo = mv.promotion.map(|p| match p {
                        Piece::Queen => "q",
                        Piece::Rook => "r",
                        Piece::Bishop => "b",
                        Piece::Knight => "n",
                        _ => "",
                    }).unwrap_or("");
                    ui.label(format!(
                        "{}. {}{}{}{}",
                        i / 2 + 1,
                        if i % 2 == 0 { "W: " } else { "B: " },
                        mv.from,
                        mv.to,
                        promo
                    ));
                }
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            let available = ui.available_size();
            let board_size = available.x.min(available.y);
            let cell_size = board_size / 8.0;

            let board_rect = egui::Rect::from_min_size(
                ui.cursor().min,
                egui::vec2(board_size, board_size),
            );

            let painter = ui.painter_at(board_rect);

            // Draw squares
            for rank in 0..8u8 {
                for file in 0..8u8 {
                    let sq = Square::new(
                        File::index(file as usize),
                        Rank::index(rank as usize),
                    );

                    // Board displayed with rank 8 at top
                    let display_rank = 7 - rank;
                    let rect = egui::Rect::from_min_size(
                        board_rect.min
                            + egui::vec2(file as f32 * cell_size, display_rank as f32 * cell_size),
                        egui::vec2(cell_size, cell_size),
                    );

                    let is_light = (file + rank) % 2 == 1;
                    let mut sq_color = if is_light {
                        egui::Color32::from_rgb(240, 217, 181)
                    } else {
                        egui::Color32::from_rgb(181, 136, 99)
                    };

                    if Some(sq) == self.selected_square {
                        sq_color = egui::Color32::from_rgb(130, 170, 80);
                    } else if self.legal_move_targets.contains(&sq) {
                        sq_color = if is_light {
                            egui::Color32::from_rgb(170, 200, 120)
                        } else {
                            egui::Color32::from_rgb(120, 160, 80)
                        };
                    }

                    painter.rect_filled(rect, 0.0, sq_color);

                    // Draw piece
                    if let Some(piece) = game_snapshot.board.piece_on(sq) {
                        let piece_color = if game_snapshot.board.colors(Color::White).has(sq) {
                            Color::White
                        } else {
                            Color::Black
                        };
                        painter.text(
                            rect.center(),
                            egui::Align2::CENTER_CENTER,
                            Self::piece_char(piece, piece_color),
                            egui::FontId::proportional(cell_size * 0.7),
                            egui::Color32::BLACK,
                        );
                    }

                    // File labels on bottom row
                    if rank == 0 {
                        let file_char = (b'a' + file) as char;
                        painter.text(
                            rect.min + egui::vec2(3.0, cell_size - 14.0),
                            egui::Align2::LEFT_BOTTOM,
                            file_char.to_string(),
                            egui::FontId::proportional(12.0),
                            if is_light {
                                egui::Color32::from_rgb(181, 136, 99)
                            } else {
                                egui::Color32::from_rgb(240, 217, 181)
                            },
                        );
                    }
                    // Rank labels on left column
                    if file == 0 {
                        painter.text(
                            rect.min + egui::vec2(3.0, 3.0),
                            egui::Align2::LEFT_TOP,
                            (rank + 1).to_string(),
                            egui::FontId::proportional(12.0),
                            if is_light {
                                egui::Color32::from_rgb(181, 136, 99)
                            } else {
                                egui::Color32::from_rgb(240, 217, 181)
                            },
                        );
                    }
                }
            }

            // Human click handling — disabled in bot-vs-bot mode
            if !self.bot_vs_bot {
                let is_human_turn = game_snapshot.side_to_move() == self.human_color
                    && !game_snapshot.is_game_over()
                    && !bot_thinking;

                if is_human_turn {
                    if let Some(pos) = ctx.input(|i| i.pointer.press_origin()) {
                        if board_rect.contains(pos) && ctx.input(|i| i.pointer.primary_pressed()) {
                            let rel = pos - board_rect.min;
                            let file = (rel.x / cell_size) as u8;
                            let rank = 7 - (rel.y / cell_size) as u8;

                            if file < 8 && rank < 8 {
                                let clicked_sq = Square::new(
                                    File::index(file as usize),
                                    Rank::index(rank as usize),
                                );

                                if let Some(from) = self.selected_square {
                                    if self.legal_move_targets.contains(&clicked_sq) {
                                        let promotion = {
                                            let state = self.shared.lock().unwrap();
                                            let is_pawn =
                                                state.game.board.piece_on(from) == Some(Piece::Pawn);
                                            let back_rank = match self.human_color {
                                                Color::White => Rank::Eighth,
                                                Color::Black => Rank::First,
                                            };
                                            if is_pawn && clicked_sq.rank() == back_rank {
                                                Some(Piece::Queen)
                                            } else {
                                                None
                                            }
                                        };

                                        let mv = Move { from, to: clicked_sq, promotion };
                                        let _ = self.move_sender.send(mv);
                                        self.selected_square = None;
                                        self.legal_move_targets.clear();
                                    } else {
                                        let has_own_piece = {
                                            let state = self.shared.lock().unwrap();
                                            state.game.board.colors(self.human_color).has(clicked_sq)
                                        };
                                        if has_own_piece {
                                            self.selected_square = Some(clicked_sq);
                                            self.update_legal_targets(clicked_sq);
                                        } else {
                                            self.selected_square = None;
                                            self.legal_move_targets.clear();
                                        }
                                    }
                                } else {
                                    let has_own_piece = {
                                        let state = self.shared.lock().unwrap();
                                        state.game.board.colors(self.human_color).has(clicked_sq)
                                    };
                                    if has_own_piece {
                                        self.selected_square = Some(clicked_sq);
                                        self.update_legal_targets(clicked_sq);
                                    }
                                }
                            }
                        }
                    }
                } else {
                    self.selected_square = None;
                    self.legal_move_targets.clear();
                }
            }
        });
    }
}

fn run_game_loop(
    shared: Arc<Mutex<SharedState>>,
    move_receiver: std::sync::mpsc::Receiver<Move>,
    nn_bot: Option<NnBot>,
    move_delay_ms: u64,
) {
    let spicy = BaselineBot::default();
    let bot_vs_bot = nn_bot.is_some();

    loop {
        let (side, is_over) = {
            let state = shared.lock().unwrap();
            (state.game.side_to_move(), state.game.is_game_over())
        };

        if is_over {
            let mut state = shared.lock().unwrap();
            state.bot_thinking = false;
            state.status_message = match state.game.outcome() {
                Some(Outcome::Checkmate { winner }) => {
                    if bot_vs_bot {
                        format!(
                            "{} wins by checkmate!",
                            if winner == Color::White { "NnBot" } else { "BaselineBot" }
                        )
                    } else {
                        format!(
                            "{} wins by checkmate!",
                            if winner == Color::White { "White" } else { "Black" }
                        )
                    }
                }
                Some(Outcome::Draw) => "Draw!".to_string(),
                None => "Game over.".to_string(),
            };
            break;
        }

        if bot_vs_bot {
            let bot_name = if side == Color::White { "NnBot" } else { "BaselineBot" };

            {
                let mut state = shared.lock().unwrap();
                state.bot_thinking = true;
                state.status_message = format!("{} thinking...", bot_name);
            }

            let game_snapshot = shared.lock().unwrap().game.clone();

            thread::sleep(Duration::from_millis(move_delay_ms));

            let mv = if side == Color::White {
                nn_bot.as_ref().unwrap().choose_move(&game_snapshot)
            } else {
                spicy.choose_move(&game_snapshot)
            };

            match mv {
                Some(mv) => {
                    let promo = mv.promotion.map(|p| match p {
                        Piece::Queen => "q",
                        Piece::Rook => "r",
                        Piece::Bishop => "b",
                        Piece::Knight => "n",
                        _ => "",
                    }).unwrap_or("");
                    let mut state = shared.lock().unwrap();
                    state.game.make_move(mv);
                    state.bot_thinking = false;
                    state.status_message =
                        format!("{} played {}{}{}", bot_name, mv.from, mv.to, promo);
                }
                None => {
                    let winner = !side;
                    let mut state = shared.lock().unwrap();
                    state.bot_thinking = false;
                    state.status_message = format!(
                        "{} wins (opponent resigned)",
                        if winner == Color::White { "NnBot" } else { "BaselineBot" }
                    );
                    break;
                }
            }
        } else {
            // Human (White) vs BaselineBot (Black)
            if side == Color::White {
                {
                    let mut state = shared.lock().unwrap();
                    state.status_message = "White (you) to move".to_string();
                    state.bot_thinking = false;
                }

                match move_receiver.recv() {
                    Ok(mv) => {
                        let mut state = shared.lock().unwrap();
                        state.game.make_move(mv);
                    }
                    Err(_) => break,
                }
            } else {
                {
                    let mut state = shared.lock().unwrap();
                    state.bot_thinking = true;
                    state.status_message = "Bot thinking...".to_string();
                }

                let game_snapshot = shared.lock().unwrap().game.clone();

                if let Some(mv) = spicy.choose_move(&game_snapshot) {
                    let mut state = shared.lock().unwrap();
                    state.game.make_move(mv);
                    state.bot_thinking = false;
                    state.status_message = format!("Bot played: {}{}", mv.from, mv.to);
                }
            }
        }
    }
}

fn main() -> eframe::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    // Usage: gui [model.onnx] [--delay MS]
    let mut nn_path: Option<PathBuf> = None;
    let mut move_delay_ms: u64 = 600;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--delay" => {
                if let Some(val) = args.get(i + 1).and_then(|s| s.parse().ok()) {
                    move_delay_ms = val;
                    i += 1;
                }
            }
            arg if !arg.starts_with('-') && nn_path.is_none() => {
                nn_path = Some(PathBuf::from(arg));
            }
            _ => {}
        }
        i += 1;
    }

    let bot_vs_bot = nn_path.is_some();

    let shared = Arc::new(Mutex::new(SharedState::new()));
    let (tx, rx) = std::sync::mpsc::channel::<Move>();

    let shared_clone = shared.clone();
    thread::spawn(move || {
        let nn_bot = if let Some(path) = nn_path {
            match NnBot::load(&path) {
                Ok(b) => {
                    println!("Loaded NnBot from {}", path.display());
                    Some(b)
                }
                Err(e) => {
                    eprintln!("Failed to load NnBot: {e}");
                    shared_clone.lock().unwrap().status_message =
                        format!("Failed to load model: {e}");
                    return;
                }
            }
        } else {
            None
        };

        run_game_loop(shared_clone, rx, nn_bot, move_delay_ms);
    });

    let title = if bot_vs_bot {
        "Chess Challenge — NnBot vs BaselineBot"
    } else {
        "Chess Challenge"
    };

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([800.0, 600.0])
            .with_title(title),
        ..Default::default()
    };

    eframe::run_native(
        title,
        options,
        Box::new(|_cc| Ok(Box::new(ChessApp::new(shared, tx, bot_vs_bot)))),
    )
}
