//+------------------------------------------------------------------+
//|                              SK10_Gold_APISignal.mq5             |
//|   Version 4.1 - Signal API + gestion du risque anti-grosses pertes|
//+------------------------------------------------------------------+
//| Nouveautés v4.1 :                                                |
//|  - Perte max journalière : on coupe et on arrête jusqu'à demain  |
//|  - Cooldown après un panier perdant (pas de re-entrée même sens) |
//|  - Pause après N paniers perdants d'affilée                      |
//|  - Filtre ADX : pas de grille ni de renfort contre une tendance  |
//|    forte                                                         |
//|  - Écart minimum entre positions (l'ATR M1 peut être minuscule)  |
//|  - Heures sans trading (rollover) + panneau d'état sur le graph  |
//| Nouveautés v4.0 :                                                |
//|  - Signal anticipé (ex. SELL sur une zone pendant que les        |
//|    bougies sont vertes) : la grille attend qu'une bougie M5      |
//|    confirme le retournement avant d'ouvrir                       |
//|  - Confirmation : N signaux successifs identiques avant d'agir    |
//|  - Signal invalide (API en erreur / démarrage / trop ancien)     |
//|    = NEUTRE (avant : probability null -> 0 -> faux SELL)         |
//|  - Signal opposé confirmé : la grille arrête d'ajouter, sort au   |
//|    break-even si possible, sinon coupe après X minutes           |
//|  - Stop loss panier en USD (avant : seulement -15% du compte)    |
//|  - Direction retrouvée depuis les positions après redémarrage    |
//+------------------------------------------------------------------+
#property copyright "Créé sur mesure"
#property version   "4.10"
#property strict

#include <Trade\Trade.mqh>
#include <Trade\SymbolInfo.mqh>

CTrade         trade;
CSymbolInfo    symbolInfo;

//--- PARAMÈTRES DE L'API SIGNAL ---
input group "=== Signal API Externe ==="
input string   SignalAPI_URL      = "https://goldstark.onrender.com/signal"; // URL de l'API
input int      SignalUpdateSec    = 20;       // Intervalle de mise à jour (secondes)
input double   BuyThreshold       = 0.55;     // Seuil de probabilité pour BUY (modèle réentraîné du 11/10/2026)
input double   SellThreshold      = 0.45;     // Seuil de probabilité pour SELL (en dessous)
input int      ConfirmReadings    = 1;        // Nb de signaux successifs identiques pour confirmer (1 = immédiat)
input int      MaxSignalAgeMin    = 5;        // Signal plus vieux que ça = NEUTRE (API en panne / endormie)
input bool     StrictSignalMode   = false;    // true = API prime toujours / false = grille verrouillée (recommandé)

//--- CONFIRMATION PAR LE PRIX (signal anticipé -> on attend que le prix tourne) ---
input group "=== Confirmation par le prix avant d'ouvrir la grille ==="
input bool            UsePriceConfirmation = true;      // Attendre que les bougies confirment le signal de l'API
input ENUM_TIMEFRAMES ConfirmTF            = PERIOD_M5; // Unité de temps des bougies de confirmation
input int             ConfirmBars          = 1;         // Nb de bougies fermées dans le sens du signal
input bool            ConfirmBreakPrevBar  = true;      // La bougie doit aussi casser le bas (SELL) / haut (BUY) de la précédente
input int             SignalWaitMin        = 60;        // Pas de confirmation après X min -> on laisse passer ce signal
input bool            ReconfirmEachGrid    = true;      // true = chaque nouvelle grille attend ses bougies de confirmation / false = un signal confirmé reste valable

//--- PARAMÈTRES DE LA GRILLE (GRID) ---
input group "=== Paramètres de Grille ==="
input double   InitialLot         = 0.01;     // Lot de départ
input double   LotIncrement       = 0.01;     // Incrément lot (DCA progressif, PAS martingale)
input double   MaxLot             = 0.10;     // Lot max par position (garde-fou)
input int      MaxOrdersPerSide   = 6;        // Max de positions par côté
input double   ATR_Multiplier     = 1.0;      // Multiplicateur ATR pour l'espacement grille
input int      ATR_Period         = 14;       // Période ATR

//--- PARAMÈTRES DE SORTIE (PROFIT & RISQUE) ---
input group "=== Prise de Profit & Sécurité ==="
input double   TargetProfitUSD    = 2.0;      // Basket TP (en devise du compte)
input double   MaxDrawdownPercent = 15.0;     // Drawdown Max (%) avant coupure
input int      MaxSpreadPoints    = 200;      // Spread Max toléré (points)
input int      TP_Points          = 1500;     // TP individuel (points) - XAUUSD : 1500 points = 15 USD de prix
input int      MaxTradeSeconds    = 30;       // Time-out par position en profit (secondes)
input ulong    MagicNumber        = 20250101; // Identifiant EA

//--- SORTIES QUAND LE SIGNAL RATE ---
input group "=== Sortie si le signal rate ==="
input double   BasketStopLossUSD  = 40.0;     // Perte max du panier en USD avant coupure (0 = désactivé)
input bool     ExitOnOppositeSignal = true;   // Signal opposé confirmé -> on arrête la grille et on sort
input int      OppositeGraceMin   = 15;       // Après signal opposé : attente max du break-even (minutes), puis coupure
input double   BreakevenExitUSD   = 0.50;     // Pendant l'attente : on ferme dès que le panier vaut >= ce montant
input int      MaxBasketMinutes   = 0;        // Panier ouvert depuis plus de X min -> on n'ajoute plus, sortie au break-even (0 = off)

//--- PROTECTIONS ANTI-GROSSES PERTES ---
input group "=== Protections anti-grosses pertes ==="
input double          DailyMaxLossUSD      = 80.0;       // Perte max sur la journée -> on ferme et on arrête jusqu'à demain (0 = off)
input int             CooldownAfterLossMin = 30;         // Après un panier perdant : pas de nouvelle grille dans le même sens pendant X min
input int             MaxLossStreak        = 2;          // Nb de paniers perdants d'affilée avant une pause (0 = off)
input int             PauseAfterStreakMin  = 120;        // Durée de la pause (minutes)
input bool            UseADXFilter         = true;       // Pas de grille CONTRE une tendance forte (ADX)
input ENUM_TIMEFRAMES ADX_TF               = PERIOD_M15; // Unité de temps de l'ADX
input double          ADX_Max              = 30.0;       // ADX au-dessus = tendance forte
input double          MinGridStepUSD       = 1.0;        // Écart minimum entre 2 positions (en $ de prix), même si l'ATR est petit
input int             NoTradeStartHour     = -1;         // Pas de nouvelle grille à partir de cette heure serveur (-1 = off)
input int             NoTradeEndHour       = -1;         // ... jusqu'à cette heure serveur (ex. 21 -> 1 = rollover)
input bool            ShowPanel            = true;       // Afficher l'état du robot sur le graphique

//--- VARIABLES GLOBALES ---
double   startingBalance     = 0.0;
int      atrHandle           = INVALID_HANDLE;
datetime lastSignalTime      = 0;
string   lockedDirection     = "";          // "BUY", "SELL" ou "" (aucune grille en cours)

string   confirmedSignal     = "NEUTRAL";   // Signal confirmé utilisé pour trader
double   confirmedProb       = 0.5;
string   pendingSignal       = "";          // Signal en cours de confirmation
int      pendingCount        = 0;
string   lastSignalStamp     = "";          // Timestamp du dernier signal compté
datetime lastSignalStampGMT  = 0;           // Heure UTC du dernier signal valide
datetime oppositeSince       = 0;           // Début du signal opposé à la grille
string   waitingSignal       = "";          // Signal reçu, en attente de confirmation par le prix
datetime waitingSince        = 0;
bool     waitingExpired      = false;
string   priceConfirmedSignal = "";         // Signal déjà confirmé par le prix (valable tant que l'API ne change pas)

int      adxHandle           = INVALID_HANDLE;
int      currentDay          = -1;
double   dayStartEquity      = 0.0;
bool     dailyStopped        = false;          // perte journalière atteinte
int      lossStreak          = 0;              // paniers perdants d'affilée
datetime pauseUntil          = 0;              // pause après série de pertes
string   lastLossDir         = "";
datetime lastLossTime        = 0;
string   blockReason         = "";             // pourquoi aucune grille n'est ouverte (affichage)
bool     addBlockedByTrend   = false;

//+------------------------------------------------------------------+
//| INITIALISATION                                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   trade.SetExpertMagicNumber(MagicNumber);
   trade.SetTypeFilling(ORDER_FILLING_IOC);
   trade.SetDeviationInPoints(30);
   symbolInfo.Name(_Symbol);

   startingBalance = AccountInfoDouble(ACCOUNT_BALANCE);

   PrintFormat("EA démarré | Symbole: %s | ContractSize: %.4f | TickValue: %.4f | VolumeStep: %.4f | MinLot: %.2f | Point: %.6f",
               _Symbol,
               SymbolInfoDouble(_Symbol, SYMBOL_TRADE_CONTRACT_SIZE),
               SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE),
               SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP),
               SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN),
               _Point);

   atrHandle = iATR(_Symbol, PERIOD_M1, ATR_Period);
   if(atrHandle == INVALID_HANDLE)
     {
      Print("Erreur init ATR");
      return(INIT_FAILED);
     }
   adxHandle = iADX(_Symbol, ADX_TF, 14);
   if(adxHandle == INVALID_HANDLE)
     {
      Print("Erreur init ADX");
      return(INIT_FAILED);
     }

   Print("N'oubliez pas d'ajouter l'URL dans Outils > Options > Expert Advisors > Autoriser les URL web : ", SignalAPI_URL);
   Print("Mode : ", StrictSignalMode ? "STRICT (API prime toujours)" : "VERROUILLÉ (grille garde sa direction)");
   PrintFormat("Confirmation prix = %s %s | stop panier = %.2f USD",
               UsePriceConfirmation ? "oui" : "non", EnumToString(ConfirmTF), BasketStopLossUSD);

   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| DEINIT                                                           |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   if(atrHandle != INVALID_HANDLE) IndicatorRelease(atrHandle);
   if(adxHandle != INVALID_HANDLE) IndicatorRelease(adxHandle);
   Comment("");
  }

//+------------------------------------------------------------------+
//| TICK PRINCIPAL                                                   |
//+------------------------------------------------------------------+
void OnTick()
  {
   symbolInfo.RefreshRates();
   double ask = symbolInfo.Ask();
   double bid = symbolInfo.Bid();
   if(ask <= 0 || bid <= 0) return;

   // 1) Mettre à jour le signal API (toutes les X secondes)
   if(TimeCurrent() - lastSignalTime >= SignalUpdateSec)
     {
      GetSignalFromAPI();
      lastSignalTime = TimeCurrent();
     }
   string signal = ActiveSignal();
   // Le signal de l'API a changé (même brièvement) : l'ancienne confirmation par le prix ne vaut plus
   if(signal != priceConfirmedSignal) priceConfirmedSignal = "";

   UpdateDay();
   double floatingProfit = GetTotalFloatingProfit();
   int    totalPositions = CountAllPositions();
   double currentEquity  = AccountInfoDouble(ACCOUNT_EQUITY);
   if(ShowPanel) UpdatePanel(signal, floatingProfit, totalPositions, currentEquity - dayStartEquity);

   // 2) Perte journalière max : on ferme tout et on ne trade plus jusqu'à demain
   if(DailyMaxLossUSD > 0 && !dailyStopped && currentEquity - dayStartEquity <= -DailyMaxLossUSD)
     {
      PrintFormat("PERTE JOURNALIÈRE MAX atteinte (%.2f USD) : fermeture et arrêt jusqu'à demain",
                  currentEquity - dayStartEquity);
      if(totalPositions > 0) CloseBasket("perte journalière", floatingProfit);
      dailyStopped = true;
      return;
     }

   // 3) Filtre spread
   int currentSpread = (int)((ask - bid) / _Point);
   if(currentSpread > MaxSpreadPoints) return;

   // 4) Drawdown global (coupure d'urgence)
   if(startingBalance > 0)
     {
      double drawdown = ((startingBalance - currentEquity) / startingBalance) * 100.0;
      if(drawdown >= MaxDrawdownPercent)
        {
         Print("ALERTE : Max Drawdown atteint ! Coupure d'urgence.");
         CloseBasket("drawdown max", floatingProfit);
         return;
        }
     }

   // 5) Basket Take Profit
   if(floatingProfit >= TargetProfitUSD && totalPositions > 0)
     {
      PrintFormat("Basket TP atteint : %.2f USD. Fermeture de tout.", floatingProfit);
      CloseBasket("basket TP", floatingProfit);
      return;
     }

   // 6) Basket Stop Loss : la perte du panier est plafonnée
   if(BasketStopLossUSD > 0 && totalPositions > 0 && floatingProfit <= -BasketStopLossUSD)
     {
      PrintFormat("STOP PANIER : perte %.2f USD <= -%.2f USD. Fermeture de tout.", floatingProfit, BasketStopLossUSD);
      CloseBasket("stop panier", floatingProfit);
      return;
     }

   // 7) Time-out
   CheckTimeBasedExit();

   // Comptage des positions
   int    buyCount   = 0, sellCount = 0;
   double lowestBuy  = 0.0, highestSell = 0.0;
   datetime oldestOpen = 0;
   CountPositions(buyCount, sellCount, lowestBuy, highestSell, oldestOpen);

   totalPositions = buyCount + sellCount;

   // Si plus aucune position, on déverrouille la direction
   if(totalPositions == 0 && lockedDirection != "")
     {
      PrintFormat(">>> GRILLE TERMINÉE : déverrouillage (ancienne direction = %s)", lockedDirection);
      lockedDirection   = "";
      oppositeSince     = 0;
      lossStreak        = 0;   // grille finie par les TP / time-out = panier gagnant
      addBlockedByTrend = false;
     }

   // Après un redémarrage de l'EA : on retrouve la direction depuis les positions ouvertes
   if(lockedDirection == "" && totalPositions > 0)
     {
      if(buyCount > 0 && sellCount == 0)  lockedDirection = "BUY";
      if(sellCount > 0 && buyCount == 0)  lockedDirection = "SELL";
      if(lockedDirection != "")
         PrintFormat(">>> Direction retrouvée depuis les positions : %s", lockedDirection);
     }

   // 8) Le signal est passé dans l'autre sens : on gèle la grille et on cherche à sortir
   if(!StrictSignalMode && ExitOnOppositeSignal && totalPositions > 0 && IsOpposite(lockedDirection, signal))
     {
      if(oppositeSince == 0)
        {
         oppositeSince = TimeCurrent();
         PrintFormat(">>> SIGNAL OPPOSÉ CONFIRMÉ (%s contre grille %s) : plus d'ajout, sortie au break-even ou dans %d min",
                     signal, lockedDirection, OppositeGraceMin);
        }
      if(floatingProfit >= BreakevenExitUSD)
        {
         PrintFormat("Sortie break-even après signal opposé : %.2f USD", floatingProfit);
         CloseBasket("break-even (signal opposé)", floatingProfit);
         return;
        }
      if(TimeCurrent() - oppositeSince >= OppositeGraceMin * 60)
        {
         PrintFormat("Signal opposé maintenu %d min : coupure du panier (%.2f USD)", OppositeGraceMin, floatingProfit);
         CloseBasket("signal opposé", floatingProfit);
         return;
        }
      return; // gel : aucune nouvelle position contre le signal
     }
   if(oppositeSince != 0 && !IsOpposite(lockedDirection, signal))
     {
      Print(">>> Le signal n'est plus opposé : la grille reprend normalement");
      oppositeSince = 0;
     }

   // 9) Panier trop ancien : on n'ajoute plus, sortie dès le break-even
   if(MaxBasketMinutes > 0 && totalPositions > 0 && oldestOpen > 0 &&
      TimeCurrent() - oldestOpen >= MaxBasketMinutes * 60)
     {
      if(floatingProfit >= BreakevenExitUSD)
        {
         PrintFormat("Panier ouvert depuis plus de %d min : sortie au break-even (%.2f USD)", MaxBasketMinutes, floatingProfit);
         CloseBasket("panier trop ancien", floatingProfit);
        }
      return;
     }

   // 10) Déterminer la direction autorisée
   bool allowBuy  = false;
   bool allowSell = false;

   if(StrictSignalMode)
     {
      // Mode strict : l'API prime toujours
      if(signal == "BUY")       allowBuy  = true;
      else if(signal == "SELL") allowSell = true;
      else                      return; // NEUTRAL = on ne trade pas
     }
   else
     {
      // Mode verrouillé (recommandé pour la grille)
      if(totalPositions > 0 && lockedDirection != "")
        {
         // Grille en cours : on continue dans la direction verrouillée
         if(lockedDirection == "BUY")  allowBuy  = true;
         if(lockedDirection == "SELL") allowSell = true;

         // Tendance forte contre la grille : on n'ajoute plus de positions (pas de renfort contre le marché)
         double adxv = 0.0;
         bool against = UseADXFilter && StrongTrendAgainst(lockedDirection, adxv);
         if(against != addBlockedByTrend)
           {
            addBlockedByTrend = against;
            if(against) PrintFormat(">>> Tendance forte contre la grille %s (ADX %.1f) : plus d'ajout", lockedDirection, adxv);
            else        Print(">>> Tendance forte terminée : la grille peut de nouveau ajouter");
           }
         if(against) { allowBuy = false; allowSell = false; }
        }
      else
        {
         // Aucune position : protections, puis l'API décide, puis on attend que le prix confirme
         string reason = NewGridBlockReason(signal);
         if(reason != blockReason)
           {
            blockReason = reason;
            if(reason != "") Print(">>> Pas de nouvelle grille : ", reason);
           }
         if(reason != "") return;
         if(!SignalReadyToTrade(signal)) return;

         if(signal == "BUY")
           {
            allowBuy        = true;
            lockedDirection = "BUY";
            PrintFormat(">>> GRILLE DÉMARRÉE : direction verrouillée = BUY (prob %.4f)", confirmedProb);
           }
         else if(signal == "SELL")
           {
            allowSell       = true;
            lockedDirection = "SELL";
            PrintFormat(">>> GRILLE DÉMARRÉE : direction verrouillée = SELL (prob %.4f)", confirmedProb);
           }
         else
           {
            return; // Signal neutre et aucune position → attendre
           }
        }
     }

   // 11) Espacement grille dynamique (ATR)
   double atrBuf[];
   if(CopyBuffer(atrHandle, 0, 0, 1, atrBuf) < 1) return;
   double atrValue = atrBuf[0];
   if(atrValue <= 0) return;
   double gridStep = MathMax(atrValue * ATR_Multiplier, MinGridStepUSD);

   // 12) Limiter le nombre de positions par côté
   if(allowBuy  && buyCount  >= MaxOrdersPerSide) allowBuy  = false;
   if(allowSell && sellCount >= MaxOrdersPerSide) allowSell = false;

   // 13) Ouverture ACHAT (grille)
   if(allowBuy)
     {
      bool open = false;
      if(buyCount == 0)
         open = true;
      else if(lowestBuy > 0 && (lowestBuy - ask) >= gridStep)
         open = true;

      if(open)
        {
         double nextLot = CalculateLotSize(buyCount);
         double tp = (TP_Points > 0) ? NormalizeDouble(ask + TP_Points * _Point, _Digits) : 0.0;
         if(trade.Buy(nextLot, _Symbol, ask, 0, tp, "Grid Buy #" + IntegerToString(buyCount + 1)))
            PrintFormat("BUY grille #%d | lot=%.2f @ %.2f | TP %.2f | step %.2f",
                        buyCount + 1, nextLot, ask, tp, gridStep);
         else
            PrintFormat("Échec BUY: %s (code %d)", trade.ResultRetcodeDescription(), trade.ResultRetcode());
        }
     }

   // 14) Ouverture VENTE (grille)
   if(allowSell)
     {
      bool open = false;
      if(sellCount == 0)
         open = true;
      else if(highestSell > 0 && (bid - highestSell) >= gridStep)
         open = true;

      if(open)
        {
         double nextLot = CalculateLotSize(sellCount);
         double tp = (TP_Points > 0) ? NormalizeDouble(bid - TP_Points * _Point, _Digits) : 0.0;
         if(trade.Sell(nextLot, _Symbol, bid, 0, tp, "Grid Sell #" + IntegerToString(sellCount + 1)))
            PrintFormat("SELL grille #%d | lot=%.2f @ %.2f | TP %.2f | step %.2f",
                        sellCount + 1, nextLot, bid, tp, gridStep);
         else
            PrintFormat("Échec SELL: %s (code %d)", trade.ResultRetcodeDescription(), trade.ResultRetcode());
        }
     }
  }

//+------------------------------------------------------------------+
//| SIGNAL ACTIF : le signal confirmé, ou NEUTRE s'il est trop vieux |
//+------------------------------------------------------------------+
string ActiveSignal()
  {
   if(lastSignalStampGMT == 0) return "NEUTRAL";
   if(TimeGMT() - lastSignalStampGMT > MaxSignalAgeMin * 60) return "NEUTRAL";
   return confirmedSignal;
  }

//+------------------------------------------------------------------+
//| Le signal anticipé de l'API est-il confirmé par les bougies ?    |
//| Ex. SELL pendant que les bougies sont vertes (prix sur une zone) |
//| -> on attend une bougie rouge qui casse le bas de la précédente  |
//+------------------------------------------------------------------+
bool SignalReadyToTrade(const string signal)
  {
   if(signal != "BUY" && signal != "SELL")
     {
      waitingSignal = ""; waitingExpired = false;
      return false;
     }
   if(!UsePriceConfirmation) return true;

   if(signal != waitingSignal)
     {
      waitingSignal  = signal;
      waitingSince   = TimeCurrent();
      waitingExpired = false;
      PrintFormat(">>> Signal %s reçu : attente de confirmation par les bougies %s (max %d min)",
                  signal, EnumToString(ConfirmTF), SignalWaitMin);
     }
   // Même signal déjà confirmé par le prix (la grille précédente s'est fermée en gain) : pas besoin de reconfirmer
   if(!ReconfirmEachGrid && signal == priceConfirmedSignal) return true;
   if(waitingExpired) return false;

   if(PriceConfirms(signal))
     {
      PrintFormat(">>> Signal %s CONFIRMÉ par le prix après %d s", signal, (int)(TimeCurrent() - waitingSince));
      if(ReconfirmEachGrid) waitingSignal = "";   // la prochaine grille devra être reconfirmée
      else                  priceConfirmedSignal = signal;
      return true;
     }

   if(SignalWaitMin > 0 && TimeCurrent() - waitingSince >= SignalWaitMin * 60)
     {
      PrintFormat(">>> Signal %s non confirmé après %d min : on le laisse passer", signal, SignalWaitMin);
      waitingExpired = true;
     }
   return false;
  }

bool PriceConfirms(const string signal)
  {
   MqlRates r[];
   ArraySetAsSeries(r, true);
   // à partir de 1 = dernière bougie FERMÉE ; +1 pour avoir la précédente
   if(CopyRates(_Symbol, ConfirmTF, 1, ConfirmBars + 1, r) < ConfirmBars + 1) return false;

   for(int k = 0; k < ConfirmBars; k++)
     {
      if(signal == "SELL")
        {
         if(r[k].close >= r[k].open) return false;                          // bougie rouge
         if(ConfirmBreakPrevBar && r[k].close >= r[k + 1].low) return false; // casse le bas précédent
        }
      else
        {
         if(r[k].close <= r[k].open) return false;                           // bougie verte
         if(ConfirmBreakPrevBar && r[k].close <= r[k + 1].high) return false; // casse le haut précédent
        }
     }
   return true;
  }

bool IsOpposite(const string direction, const string signal)
  {
   return (direction == "BUY" && signal == "SELL") || (direction == "SELL" && signal == "BUY");
  }

//+------------------------------------------------------------------+
//| FONCTION : RÉCUPÉRER LE SIGNAL DEPUIS L'API                      |
//+------------------------------------------------------------------+
void GetSignalFromAPI()
  {
   string method  = "GET";
   string url     = SignalAPI_URL;
   string headers = "Content-Type: application/json\r\n";
   int    timeout = 5000;
   char   data[];
   char   result[];
   string result_headers;

   ResetLastError();
   int res = WebRequest(method, url, headers, timeout, data, result, result_headers);

   if(res == -1)
     {
      int err = GetLastError();
      PrintFormat("Erreur WebRequest (%d). Vérifiez que l'URL est autorisée dans Outils > Options > Expert Advisors.", err);
      return;
     }

   if(res != 200)
     {
      PrintFormat("Erreur HTTP %d lors de la requête API.", res);
      return;
     }

   string json = CharArrayToString(result);

   // API en erreur / en démarrage : on ne compte pas ce signal
   string status = ExtractJSONValue(json, "status");
   if(status != "ready")
     {
      PrintFormat("API pas prête (status=%s) : signal ignoré", status);
      return;
     }

   string probStr  = ExtractJSONValue(json, "probability");
   string stamp    = ExtractJSONValue(json, "timestamp");

   // null ou absent : surtout NE PAS le lire comme 0 (= faux SELL)
   if(probStr == "" || probStr == "null" || stamp == "" || stamp == "null")
     {
      Print("Probabilité absente (API en démarrage ?) : signal ignoré");
      return;
     }

   datetime stampGMT = IsoToTime(stamp);
   if(stampGMT == 0 || TimeGMT() - stampGMT > MaxSignalAgeMin * 60)
     {
      PrintFormat("Signal trop ancien (%s) : NEUTRE", stamp);
      return;
     }
   lastSignalStampGMT = stampGMT;

   if(stamp == lastSignalStamp) return;   // déjà compté, pas un nouveau calcul
   lastSignalStamp = stamp;

   double probability = StringToDouble(probStr);
   string raw = "NEUTRAL";
   if(probability >= BuyThreshold)       raw = "BUY";
   else if(probability <= SellThreshold) raw = "SELL";

   if(raw == pendingSignal) pendingCount++;
   else { pendingSignal = raw; pendingCount = 1; }

   PrintFormat(">>> SIGNAL API : %s (prob %.4f) | confirmation %d/%d | signal actif = %s",
               raw, probability, MathMin(pendingCount, ConfirmReadings), ConfirmReadings, confirmedSignal);

   if(pendingCount >= ConfirmReadings && confirmedSignal != raw)
     {
      confirmedSignal = raw;
      confirmedProb   = probability;
      PrintFormat(">>> SIGNAL CONFIRMÉ : %s (prob %.4f)", confirmedSignal, confirmedProb);
     }
  }

//--- "2026-10-08T13:18:01Z" -> datetime (UTC)
datetime IsoToTime(string iso)
  {
   StringReplace(iso, "-", ".");
   StringReplace(iso, "T", " ");
   StringReplace(iso, "Z", "");
   return StringToTime(iso);
  }

//+------------------------------------------------------------------+
//| FONCTION : EXTRAIRE UNE VALEUR D'UN JSON SIMPLE                  |
//+------------------------------------------------------------------+
string ExtractJSONValue(const string json, const string key)
  {
   string searchKey = "\"" + key + "\":";
   int startPos = StringFind(json, searchKey);
   if(startPos == -1) return "";

   startPos += StringLen(searchKey);

   while(startPos < StringLen(json) && StringGetCharacter(json, startPos) == ' ')
      startPos++;

   if(startPos >= StringLen(json)) return "";

   if(StringGetCharacter(json, startPos) == '"')
     {
      startPos++;
      int endPos = StringFind(json, "\"", startPos);
      if(endPos == -1) return "";
      return StringSubstr(json, startPos, endPos - startPos);
     }
   else
     {
      int endPos = startPos;
      while(endPos < StringLen(json) &&
            StringGetCharacter(json, endPos) != ',' &&
            StringGetCharacter(json, endPos) != '}' &&
            StringGetCharacter(json, endPos) != ']')
        {
         endPos++;
        }
      return StringSubstr(json, startPos, endPos - startPos);
     }
  }

//+------------------------------------------------------------------+
//| FONCTIONS AUXILIAIRES                                            |
//+------------------------------------------------------------------+

//--- Profit flottant net de l'EA
double GetTotalFloatingProfit()
  {
   double total = 0.0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetInteger(POSITION_MAGIC) != (long)MagicNumber) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;

      total += PositionGetDouble(POSITION_PROFIT);
      total += PositionGetDouble(POSITION_SWAP);
      total += PositionGetDouble(POSITION_COMMISSION);
     }
   return total;
  }

//--- Compte toutes les positions de l'EA
int CountAllPositions()
  {
   int c = 0;
   for(int i = 0; i < PositionsTotal(); i++)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetInteger(POSITION_MAGIC) != (long)MagicNumber) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      c++;
     }
   return c;
  }

//--- Comptage, extrêmes et plus ancienne ouverture
void CountPositions(int &buys, int &sells, double &lowBuy, double &highSell, datetime &oldest)
  {
   buys = 0; sells = 0;
   lowBuy = 0.0; highSell = 0.0;
   oldest = 0;

   for(int i = 0; i < PositionsTotal(); i++)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetInteger(POSITION_MAGIC) != (long)MagicNumber) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;

      double   price  = PositionGetDouble(POSITION_PRICE_OPEN);
      long     type   = PositionGetInteger(POSITION_TYPE);
      datetime opened = (datetime)PositionGetInteger(POSITION_TIME);
      if(oldest == 0 || opened < oldest) oldest = opened;

      if(type == POSITION_TYPE_BUY)
        {
         if(buys == 0 || price < lowBuy) lowBuy = price;
         buys++;
        }
      else if(type == POSITION_TYPE_SELL)
        {
         if(sells == 0 || price > highSell) highSell = price;
         sells++;
        }
     }
  }

//--- Ferme le panier, remet la grille à zéro et compte les paniers perdants
void CloseBasket(const string reason, const double pnl)
  {
   string dir = lockedDirection;
   CloseAllPositions();
   startingBalance   = AccountInfoDouble(ACCOUNT_BALANCE);
   lockedDirection   = "";
   oppositeSince     = 0;
   addBlockedByTrend = false;

   if(pnl < 0)
     {
      lossStreak++;
      priceConfirmedSignal = "";   // après une perte, le prix devra reconfirmer
      lastLossDir  = dir;
      lastLossTime = TimeCurrent();
      PrintFormat("Panier %s fermé en PERTE (%s) : %.2f USD | pertes d'affilée = %d", dir, reason, pnl, lossStreak);
      if(MaxLossStreak > 0 && lossStreak >= MaxLossStreak)
        {
         pauseUntil = TimeCurrent() + PauseAfterStreakMin * 60;
         PrintFormat(">>> %d paniers perdants d'affilée : PAUSE de %d min", lossStreak, PauseAfterStreakMin);
         lossStreak = 0;
        }
     }
   else
     {
      lossStreak = 0;
      PrintFormat("Panier %s fermé (%s) : %+.2f USD", dir, reason, pnl);
     }
  }

//+------------------------------------------------------------------+
//| PROTECTIONS                                                      |
//+------------------------------------------------------------------+

//--- Nouveau jour : on repart avec une nouvelle limite de perte journalière
void UpdateDay()
  {
   int day = (int)(TimeCurrent() / 86400);
   if(day == currentDay) return;
   currentDay     = day;
   dayStartEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   if(dailyStopped) Print(">>> Nouveau jour : le robot peut de nouveau trader");
   dailyStopped   = false;
  }

//--- Raison d'empêcher une NOUVELLE grille ("" = autorisé)
string NewGridBlockReason(const string signal)
  {
   if(signal != "BUY" && signal != "SELL") return "";
   if(dailyStopped)
      return StringFormat("perte journalière max atteinte (%.0f USD), reprise demain", DailyMaxLossUSD);
   if(TimeCurrent() < pauseUntil)
      return StringFormat("pause après pertes, reprise à %s", TimeToString(pauseUntil, TIME_MINUTES));
   if(CooldownAfterLossMin > 0 && signal == lastLossDir &&
      TimeCurrent() - lastLossTime < CooldownAfterLossMin * 60)
      return StringFormat("cooldown après un panier %s perdant, reprise à %s",
                          signal, TimeToString(lastLossTime + CooldownAfterLossMin * 60, TIME_MINUTES));
   if(InNoTradeHours())
      return StringFormat("heures sans trading (%dh -> %dh serveur)", NoTradeStartHour, NoTradeEndHour);
   double adxv = 0.0;
   if(UseADXFilter && StrongTrendAgainst(signal, adxv))
      return StringFormat("tendance forte contre le %s (ADX %s = %.1f)", signal, EnumToString(ADX_TF), adxv);
   return "";
  }

//--- Tendance forte dans le sens contraire ? (ADX haut + DI dominant opposé, bougie fermée)
bool StrongTrendAgainst(const string direction, double &adxValue)
  {
   double a[], p[], m[];
   if(CopyBuffer(adxHandle, 0, 1, 1, a) < 1 || CopyBuffer(adxHandle, 1, 1, 1, p) < 1 ||
      CopyBuffer(adxHandle, 2, 1, 1, m) < 1)
      return false;
   adxValue = a[0];
   if(a[0] < ADX_Max) return false;
   if(direction == "SELL" && p[0] > m[0]) return true;   // forte hausse
   if(direction == "BUY"  && m[0] > p[0]) return true;   // forte baisse
   return false;
  }

bool InNoTradeHours()
  {
   if(NoTradeStartHour < 0 || NoTradeEndHour < 0) return false;
   MqlDateTime t;
   TimeToStruct(TimeCurrent(), t);
   if(NoTradeStartHour <= NoTradeEndHour)
      return t.hour >= NoTradeStartHour && t.hour < NoTradeEndHour;
   return t.hour >= NoTradeStartHour || t.hour < NoTradeEndHour;   // ex. 21h -> 1h
  }

//--- Panneau d'état sur le graphique
void UpdatePanel(const string signal, const double floating, const int positions, const double dayPnL)
  {
   string status = "prêt";
   if(positions > 0)
      status = StringFormat("grille %s : %d position(s), %+.2f $%s%s", lockedDirection, positions, floating,
                            oppositeSince > 0 ? " | SIGNAL OPPOSÉ : sortie en cours" : "",
                            addBlockedByTrend ? " | tendance forte : plus d'ajout" : "");
   else if(blockReason != "")
      status = "bloqué : " + blockReason;
   else if(waitingSignal != "")
      status = waitingExpired ? StringFormat("signal %s non confirmé, ignoré", waitingSignal)
                              : StringFormat("attente bougie %s qui confirme le %s", EnumToString(ConfirmTF), waitingSignal);

   Comment(StringFormat("SK10 Gold v4.1\n"
                        "Signal API : %s (prob %.3f)\n"
                        "État : %s\n"
                        "Jour : %+.2f $  (limite -%.0f $)%s\n"
                        "Paniers perdants d'affilée : %d / %d",
                        signal, confirmedProb, status, dayPnL, DailyMaxLossUSD,
                        dailyStopped ? "  ARRÊTÉ" : "", lossStreak, MaxLossStreak));
  }

//--- Ferme toutes les positions de l'EA
void CloseAllPositions()
  {
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetInteger(POSITION_MAGIC) != (long)MagicNumber) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;

      if(!trade.PositionClose(ticket))
         PrintFormat("Échec close #%d : %s", ticket, trade.ResultRetcodeDescription());
     }
  }

//--- Time-out : ferme les positions anciennes ET en profit
void CheckTimeBasedExit()
  {
   if(MaxTradeSeconds <= 0) return;
   datetime now = TimeCurrent();

   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetInteger(POSITION_MAGIC) != (long)MagicNumber) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;

      datetime openTime = (datetime)PositionGetInteger(POSITION_TIME);
      double   profit   = PositionGetDouble(POSITION_PROFIT);

      if((now - openTime) >= MaxTradeSeconds && profit > 0)
        {
         if(trade.PositionClose(ticket))
            PrintFormat("Time-out close #%d (profit %.2f)", ticket, profit);
        }
     }
  }

//--- Calcul du lot : DCA progressif plafonné (PAS martingale)
double CalculateLotSize(int count)
  {
   double lot = InitialLot + (count * LotIncrement);
   if(lot > MaxLot) lot = MaxLot;

   double step   = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);

   if(step <= 0) step = 0.01;

   lot = MathFloor(lot / step) * step;

   if(lot < minLot) lot = minLot;
   if(lot > maxLot) lot = maxLot;

   return NormalizeDouble(lot, 2);
  }
//+------------------------------------------------------------------+
