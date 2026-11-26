"""Fixed text content for the synthetic corpus (kept out of the generator).

Content is authored here so diffs to ground truth are reviewable in one
place. Arabic strings are stored in logical order.
"""

LEFT_COL_A = (
    "The river cleanup drew eighty volunteers last Saturday and filled forty five "
    "bags with debris. Organizers thanked the boating club for ferry support."
)
LEFT_COL_B = (
    "Library hours extend to nine in the evening on weekdays starting next month. "
    "Weekend hours remain unchanged through the winter season."
)
LEFT_COL_C = (
    "Found: a silver bracelet near the ferry terminal. Describe the engraving to "
    "claim it at the harbor office during opening hours."
)
RIGHT_COL_A = (
    "The volunteer schedule for the park cleanup now includes a weekday shift. "
    "Sign up through the community portal before Thursday evening."
)
RIGHT_COL_B = (
    "Farmers market vendors report record attendance this season. Stall applications "
    "for the spring market open on the first Monday of December."
)
RIGHT_COL_C = (
    "Lost: a black umbrella with a wooden handle, last seen at the library reading "
    "room. A small reward is offered for its return."
)

BROCHURE_COL_1 = (
    "Starter tier includes ten thousand requests per month, community support and "
    "a single shared workspace for small evaluation projects."
)
BROCHURE_COL_2 = (
    "Growth tier adds priority queues, five workspaces, audit logs and ninety day "
    "retention for all extraction and retrieval telemetry."
)
BROCHURE_COL_3 = (
    "Enterprise tier provides dedicated clusters, custom retention, private "
    "connectivity and a named support engineer for onboarding."
)
BROCHURE_COL_4 = (
    "Starter is free forever and requires no credit card. Usage beyond the monthly "
    "allowance is rejected with a clear rate limit response."
)
BROCHURE_COL_5 = (
    "Growth is billed per workspace monthly. Annual prepay discounts apply "
    "automatically at checkout for qualifying teams."
)
BROCHURE_COL_6 = (
    "Enterprise pricing is quoted per contract. Contact sales with expected volume "
    "and compliance requirements for a written estimate."
)

SCAN_CONTRACT_P1 = [
    "SERVICE AGREEMENT between Harbor Logistics Ltd and Delta Analytics.",
    "1. The provider shall deliver monthly analytics reports by the fifth business day.",
    "2. Fees are payable net thirty days from receipt of a valid invoice.",
    "3. Both parties shall keep all shared data strictly confidential.",
]
SCAN_CONTRACT_P2 = [
    "4. Either party may terminate with sixty days written notice.",
    "5. This agreement is governed by the laws of the Arab Republic of Egypt.",
    "Signed in Cairo on 12 September 2026 in two original counterparts.",
]

FINANCIAL_HEADER = ["Segment", "Q1", "Q2", "Q3", "Q4", "Total"]
FINANCIAL_ROWS = [
    ["Logistics", "1,102,400", "1,188,200", "1,240,750", "1,305,110", "4,836,460"],
    ["Retail", "986,300", "1,004,120", "1,120,000", "1,231,450", "4,341,870"],
    ["Wholesale", "2,340,000", "2,310,500", "2,402,300", "2,511,800", "9,564,600"],
    ["Digital", "412,800", "488,150", "530,600", "612,900", "2,044,450"],
    ["Services", "733,200", "760,900", "802,400", "851,640", "3,148,140"],
    ["Other", "96,400", "102,300", "98,700", "107,900", "405,300"],
    ["COGS", "-2,980,000", "-3,010,500", "-3,140,200", "-3,305,700", "-12,436,400"],
    ["Gross profit", "2,691,100", "2,843,670", "3,054,550", "3,315,100", "11,904,420"],
    ["Payroll", "-1,540,000", "-1,560,000", "-1,585,000", "-1,620,000", "-6,305,000"],
    ["Marketing", "-320,400", "-340,100", "-355,800", "-402,300", "-1,418,600"],
    ["Operations", "-410,200", "-430,900", "-440,600", "-470,800", "-1,752,500"],
    ["Depreciation", "-95,000", "-95,000", "-100,000", "-100,000", "-390,000"],
    ["Operating income", "325,500", "417,670", "573,150", "722,000", "2,038,320"],
    ["Interest", "-40,200", "-38,900", "-37,600", "-36,300", "-153,000"],
    ["Tax", "-71,300", "-94,600", "-133,900", "-171,400", "-471,200"],
    ["Net income", "214,000", "284,170", "401,650", "514,300", "1,414,120"],
]

SPLIT_ROWS = [
    ["A-101", "Packing tape 48mm", "Cairo North", "182"],
    ["A-102", "Bubble wrap roll", "Cairo North", "74"],
    ["B-220", "Pallet jack", "Alexandria", "6"],
    ["B-221", "Shelf bracket set", "Alexandria", "143"],
    ["C-330", "Thermal labels", "Giza", "1280"],
    ["C-331", "Printer ribbon", "Giza", "96"],
    ["D-410", "Safety gloves", "Cairo North", "340"],
    ["D-411", "Safety goggles", "Alexandria", "88"],
    ["E-500", "Forklift battery", "Giza", "12"],
    ["E-501", "Charging cable 2m", "Giza", "54"],
    ["F-612", "Dock bumper", "Alexandria", "20"],
    ["F-613", "Wheel chock", "Cairo North", "46"],
]

CLAUSE_WARRANTY = (
    "7.1 Warranty. The provider warrants that services will be performed in a "
    "professional manner consistent with industry standards. Except as stated, all "
    "implied warranties are expressly disclaimed to the fullest extent permitted."
)
CLAUSE_LIABILITY = (
    "9.2 Limitation of Liability. Neither party is liable for indirect or "
    "consequential damages. Aggregate liability under this agreement shall not exceed "
    "the fees paid during the twelve months preceding the claim."
)
CLAUSE_TERMINATION = (
    "11.3 Termination for Cause. Either party may terminate this agreement if the "
    "other party materially breaches any provision and fails to cure the breach "
    "within thirty days of written notice describing the breach."
)
CLAUSE_GOVERNING = (
    "14.1 Governing Law. This agreement is governed by the laws of the Arab Republic "
    "of Egypt, excluding conflict of law rules. The courts of Cairo have exclusive "
    "jurisdiction over any dispute arising from this agreement."
)

HANDWRITING_LINES = [
    "Buy two spare printer ribbons before Friday.",
    "Call the warehouse about the missing pallet.",
    "Reminder: audit prep meeting moved to 9:30.",
]

ARABIC_HEADING = "إشعار مهم بخصوص الصيانة الدورية"
ARABIC_PARA_1 = (
    "سيتم إجراء أعمال الصيانة الدورية لنظام التكييف يوم الثلاثاء الموافق الخامس عشر من "
    "ديسمبر من الساعة التاسعة صباحا حتى الساعة الثانية بعد الظهر."
)
ARABIC_PARA_2 = (
    "يرجى من جميع الموظفين إغلاق النوافذ وإطفاء الأجهزة غير الضرورية قبل مغادرة "
    "المكاتب في اليوم السابق لموعد الصيانة المحدد."
)
ARABIC_HEADING_2 = "تعليمات السلامة أثناء الصيانة"
ARABIC_PARA_3 = (
    "يمنع الدخول إلى منطقة العمل أثناء تنفيذ أعمال الصيانة إلا للفريق المختص وبتصريح "
    "مسبق من إدارة المرافق والخدمات العامة."
)
ARABIC_NOTICE_PERIOD = "الخامس عشر من ديسمبر"
ARABIC_TOTAL_LINE = "المبلغ الإجمالي المستحق: ٢٤٨٠٫٧٥ يورو"
ARABIC_PAYMENT_LINE = "يرجى ذكر رقم المرجع عند تنفيذ التحويل البنكي"
ARABIC_PROBE_QUERY = "موعد أعمال الصيانة الدورية لنظام التكييف"
