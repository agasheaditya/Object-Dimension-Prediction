import streamlit as st
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
from PIL import Image
from datetime import datetime
import imutils
import psycopg2
import cv2


hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
hide_streamlit_style = """
<style>
.css-hi6a2p {padding-top: 0rem;}

</style>

"""
padding = 0
st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)


## Database

try:
    connection = psycopg2.connect(user="doadmin",
                                  password="qXgoR9IVdtY85wrz",
                                  host="db-postgresql-blr1-98598-do-user-9198634-0.b.db.ondigitalocean.com",
                                  port=25060,
                                  database="defaultdb")
except (Exception, psycopg2.Error) as error:
        print("Error while connecting to cloud database!! ", error)


cursor = connection.cursor()
print(cursor)
st.title("Inventory Tracker")
st.markdown("---")
st.write("Video Feed")
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)
col1, col2 = st.columns(2)

#stop = col2.button("Stop")
run = col1.checkbox('Run')
capture = col2.button("capture")


st.markdown("---")
col1, col2, col3 = st.columns(3)

location1 = col1.selectbox("Location: ",["Select", "Pune", "Jabalpur", "Bangalore", "Shankarapalli"], key="loc1")
to_from = col2.selectbox("",["Select","To", "From"])
location2 = col3.selectbox("",["Select", "Pune", "Jabalpur", "Bangalore", "Shankarapalli"])

st.markdown("---")

col1, col2, col3 = st.columns((2,2,2))
type = col1.selectbox("SKU",["Select", "Wall Panel", "Kicker", "Vertical Pipe", "Ledger Pipe", "U Jack", "Base Jack"], key="loc2")
label = "Browse Images"
#input_image = col2.file_uploader(label, type=[".jpg", ".jpeg", ".png", ".svg"], accept_multiple_files=False)


while run:
    _, frame = camera.read()
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)
    if capture:
        cv2.imwrite("inputs/vid_op.png", frame)
        run = False
        run = True
        capture = False
#    if stop:
#        camera.release()
#        cv2.destroyAllWindows()
#        run = False
#        break
else:
    camera.release()
    cv2.destroyAllWindows()
    #st.write('Stopped')

wall_panel = {"Wall Panel": (2050, 600)}
kicker = {"Kicker-1": (1600, 130), "Kicker-2":(2000,130)}
ledger_pipe = {"Ledger Pipe - 1": 1160, "Ledger Pipe - 2": 1750}
vertical_pipe = {"Vertical Pipe - 1": 1080, "Verical Pipe - 2": 2080, "Vertical Pipe - 3":850}
U_jack = {"U Jack": 680, "U Jack With Nut": 870}
base_jack = {"Base Jack - 1": 760, "Base Jack - 2": 870}


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

st.markdown("---")

col1, col2, col3 = st.columns((2,2,2))

col3.markdown("##### Identified Dimensions")
#st.markdown(" ---")
input_image = cv2.imread("inputs/vid_op.png")

if input_image is not None and type != "Select":
    #image = Image.open(input_image)
    image = input_image #np.array(image)
    scale_percent = 20  # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #gray = cv2.GaussianBlur(gray, (9, 9), 0)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    (T, thresh) = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = np.ones((10, 10), np.uint8)
    # opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    v = np.median(closing)

    # ---- apply automatic Canny edge detection using the computed median----
    lower = int(max(0, (1.0 - 0.33) * v))
    upper = int(min(255, (1.0 + 0.33) * v))

    edged = cv2.Canny(closing, lower, upper)

    kernel = np.ones((5, 5), np.uint8)
    dilat = cv2.dilate(edged, None, iterations=1)
    cnts = cv2.findContours(dilat.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    (cnts, _) = contours.sort_contours(cnts)
    pixelsPerMetric = None

    counter = 1
    record = []
    for c in cnts:
        # if the contour is not sufficiently large, ignore it
        if cv2.contourArea(c) < 100:
            continue

        # compute the rotated bounding box of the contour
        orig = image.copy()
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")

        box = perspective.order_points(box)
        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

        # loop over the original points and draw them
        for (x, y) in box:
            cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)

        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        # draw the midpoints on the image
        cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

        # draw lines between the midpoints
        cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                 (255, 0, 255), 2)
        cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                 (255, 0, 255), 2)

        # compute the Euclidean distance between the midpoints
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        if pixelsPerMetric is None:
            pixelsPerMetric = dB / 8  # args["width"]

        # compute the size of the object
        dimA = (dA / pixelsPerMetric) * 2.54
        dimB = (dB / pixelsPerMetric) * 2.54
        # print(dimA,"x", dimB)

        cv2.putText(orig, "{:.1f}cm".format(dimA),
                    (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)
        cv2.putText(orig, "{:.1f}cm".format(dimB),
                    (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)

        if counter == 2:
            record.append([dimA, dimB])  # length , width
            cv2.imwrite("op_0.jpg", orig)
        counter += 1

    col2.write("##")

    col1, col2 = st.columns((2,1))

    #col3.info(
    #    " Length: " + str(round(record[0][0], 2)) + " CM")
    #col4.info(" Width: " + str(round(record[0][1], 2)) + "CM")

    now = str(datetime.now())
    print("now = ", now.split(" ")[0])
    now = now.split(" ")[0]
    curr_transation_id_query = "select transactionid FROM public.inventorytransactions_cloud where location='"+location1+"' ORDER BY id DESC LIMIT 1;"
    cursor.execute(curr_transation_id_query)
    curr_transation_id = cursor.fetchall()
    new_transaction_id = curr_transation_id[0][0][:1]+str(int(curr_transation_id[0][0][1:])+1)
    print("new trans id", new_transaction_id)
    col2.markdown("##")
    if type == "Wall Panel":
        for key, val in wall_panel.items():
            # print(key, val, int(val[0]*0.8) , int(val[0]*1.2), record[0][0])
            col1.image("op_0.jpg", 512, 400)
            col2.info(key)
            col2.info(" Length: " + str(val[0]) + "MM")
            col2.info(" Width: " + str(val[1]) + " MM")

            if int(record[0][0] * 10) in range(int(val[0] * 0.9), int(val[0] * 1.1)) and int(
                    record[0][1] * 10) in range(int(val[1] * 0.9), int(val[1] * 1.1)):
                col1.image("op_0.jpg", 1080,720)
                col2.info(key)
                col2.info(" Length: " + str(val[0]) + "MM")
                col2.info(" Width: " + str(val[1]) + " MM")

                insert_q = "insert into public.inventorytransactions_cloud  \
                (transactionid, date, sku,sku_category, length, width, from_to, source_destination,location) VALUES \
                (%s, %s, 'Wall Panel', 'Mivan', 2050,600, %s, %s, %s); "

                #cursor.execute(insert_q % ("'"+new_transaction_id+"'", now.split(" ")[0],"'"+to_from+"'", "'"+location2+"'", "'"+location1+"'" )) #
                connection.commit()
                print("Data pushed successfully!")
                get_count_q = "select total from public.inventorytotal_cloud where location='"+location1+"' AND sku='Wall Panel' ;"
                cursor.execute(get_count_q)
                count = cursor.fetchall()[0][0]
                leng = val[0]
                if to_from == "From":
                    total = count + 1
                elif to_from == "To":
                    total = count - 1
                update_q = "UPDATE public.inventorytotal_cloud SET total=%s WHERE location='"+location1+"' AND length="+str(leng)+";"
                #cursor.execute(update_q % (total))
                connection.commit()
                print("total updated successfully !")

    elif type == "Kicker":
        for key, val in kicker.items():
            if int(record[0][0] * 10) in range(int(val[0] * 0.85), int(val[0] * 1.15)) and int(
                    record[0][1] * 10) in range(int(val[1] * 0.85), int(val[1] * 1.15)):
                #col3.info(
                #    " Length: " + str(val[0]) + " CM")
                #col4.info(" Width: " + str(val[1]) + " CM")
                #col3.info(key)
                #col1.image("op_0.jpg", 512, 256)
                print(key)
                col3.info(
                    " Length: " + str(val[0]) + " MM" + " Width: " + str(val[1]) + " MM")
                col2.info(key)
                col1.image("op_0.jpg", 512, 256)

                insert_q = "insert into public.inventorytransactions_cloud  \
                                (transactionid, date, sku,sku_category, length, width, from_to, source_destination,location) VALUES \
                                (%s, %s, 'Kicker', 'Mivan', 2050,600, %s, %s, %s); "
                #cursor.execute(insert_q % (
                #"'" + new_transaction_id + "'", now.split(" ")[0], "'" + to_from + "'", "'" + location2 + "'",
                #"'" + location1 + "'"))
                #connection.commit()
                print("Data pushed successfully!")
                get_count_q = "select total from public.inventorytotal_cloud where location='" + location1 + "' AND sku='Kicker' ;"
                cursor.execute(get_count_q)
                count = cursor.fetchall()[0][0]
                leng = val[0]
                if to_from == "From":
                    total = count + 1
                elif to_from == "To":
                    total = count - 1
                update_q = "UPDATE public.inventorytotal_cloud SET total=%s WHERE location='" + location1 + "' AND length=" + str(leng) + ";"
                #cursor.execute(update_q % (total))
                connection.commit()
                print("total updated successfully !")

    elif type == "Ledger Pipe":
        for key, val in ledger_pipe.items():
            if int(record[0][0] * 10) in range(int(val * 0.75), int(val * 1.25)):
                #col3.info(key)
                #col1.image("op_0.jpg", 512, 256)
                col3.info(
                    " Length: " + str(val[0]) + " MM" + " Width: " + str(val[1]) + " MM")
                col2.info(key)
                #col1.image("op_0.jpg", 512, 256)

                leng = val
                insert_q = "insert into public.inventorytransactions_cloud  \
                                (transactionid, date, sku,sku_category, length, width, from_to, source_destination,location) VALUES \
                                (%s, %s, 'Ledger Pipe', 'MS Formwork', %s ,0, %s, %s, %s); "
                cursor.execute(insert_q % (
                "'" + new_transaction_id + "'", now.split(" ")[0], leng,"'" + to_from + "'", "'" + location2 + "'",
                "'" + location1 + "'"))
                connection.commit()
                print("Data pushed successfully!")
                get_count_q = "select total from public.inventorytotal_cloud where location='" + location1 + "' AND sku='Ledger Pipe' AND length="+str(leng)+";"
                cursor.execute(get_count_q)
                count = cursor.fetchall()[0][0]
                if to_from == "From":
                    total = count + 1
                elif to_from == "To":
                    total = count - 1
                update_q = "UPDATE public.inventorytotal_cloud SET total=%s WHERE location='" + location1 + "' AND length=" + str(
                    leng) + ";"
                cursor.execute(update_q % (total))
                connection.commit()
                print("total updated successfully !")

    elif type == "Vertical Pipe":
        for key, val in vertical_pipe.items():
            if int(record[0][0] * 10) in range(int(val * 0.7), int(val * 1.3)):
                #col3.info(key)
                #col1.image("op_0.jpg", 512, 256)
                col3.info(
                    " Length: " + str(val) )
                col2.info(key)
                col1.image("op_0.jpg", 512, 256)
                leng = val
                insert_q = "insert into public.inventorytransactions_cloud  \
                                                (transactionid, date, sku,sku_category, length, width, from_to, source_destination,location) VALUES \
                                                (%s, %s, 'Vertical Pipe', 'MS Formwork', %s ,0, %s, %s, %s); "
                #cursor.execute(insert_q % (
                #    "'" + new_transaction_id + "'", now.split(" ")[0], leng, "'" + to_from + "'", "'" + location2 + "'",
                #    "'" + location1 + "'"))
                connection.commit()
                print("Data pushed successfully!")
                #get_count_q = "select total from public.inventorytotal_cloud where location='" + location1 + "' AND sku='Vertical Pipe' AND length=" + str(leng) + ";"
                #cursor.execute(get_count_q)
                #count = cursor.fetchall()[0][0]
                #if to_from == "From":
                #    total = count + 1
                #elif to_from == "To":
                #    total = count - 1
                #update_q = "UPDATE public.inventorytotal_cloud SET total=%s WHERE location='" + location1 + "' AND length=" + str( leng) + ";"
                #cursor.execute(update_q % (total))
                #connection.commit()
                print("total updated successfully !")

    elif type == "U Jack":
        for key, val in U_jack.items():
            if int(record[0][0] * 10) in range(int(val * 0.9), int(val * 1.1)):
                #col3.info( key)
                #col1.image("op_0.jpg", 512, 256)
                col3.info(
                    " Length: " + str(val[0]) + " MM" + " Width: " + str(val[1]) + " MM")
                col2.info(key)
                col1.image("op_0.jpg", 512, 256)
                leng = val
                insert_q = "insert into public.inventorytransactions_cloud  \
                            (transactionid, date, sku,sku_category, length, width, from_to, source_destination,location) VALUES \
                            (%s, %s, 'U Jack', 'MS Formwork', %s ,0, %s, %s, %s); "
                cursor.execute(insert_q % (
                    "'" + new_transaction_id + "'", now.split(" ")[0], leng, "'" + to_from + "'", "'" + location2 + "'",
                    "'" + location1 + "'"))
                connection.commit()
                print("Data pushed successfully!")
                get_count_q = "select total from public.inventorytotal_cloud where location='" + location1 + "' AND sku='U Jack' AND length=" + str(
                    leng) + ";"
                cursor.execute(get_count_q)
                count = cursor.fetchall()[0][0]
                if to_from == "From":
                    total = count + 1
                elif to_from == "To":
                    total = count - 1
                update_q = "UPDATE public.inventorytotal_cloud SET total=%s WHERE location='" + location1 + "' AND length=" + str(
                    leng) + ";"
                cursor.execute(update_q % (total))
                connection.commit()
                print("total updated successfully !")

    elif type == "Base Jack":
        for key, val in base_jack.items():
            if int(record[0][0] * 10) in range(int(val * 0.95), int(val * 1.05)):
                #col3.info(key)
                #col2.image("op_0.jpg", 512, 256)
                col3.info(
                    " Length: " + str(val[0]) + " MM" + " Width: " + str(val[1]) + " MM")
                col2.info(key)
                col1.image("op_0.jpg", 512, 256)

