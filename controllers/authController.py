from fastapi import HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi import APIRouter
from validator import validateRunAlgoRequest
from config import ParamsList
from auth import register_user as auth_register_user

from db.dbSession import databaseInstance as db
from mailer.mailSession import mailerInstance as mailer, MailMessage
from utils.otpGenerator import generateOTP
from utils.validator import *

import uuid
from models import *


apiRouter = APIRouter(prefix="/api/auth")

paramsList = ParamsList()


@apiRouter.post(
    "/register/",
    summary="Register a new user",
    description="Accepts user details and registers a new user.",
)
async def register_user(user: RegisterUserModel):

    if not db._connection_pool:
        await db.connect()

    con = await db._connection_pool.acquire()

    try:

        if not (
            validateEmail(user.userEmail)
            and validatePassword(user.password)
            and user.userName != ""
            and user.userName is not None
        ):
            raise HTTPException(status_code=400, detail="Invalid Email")

        email = await con.fetch(
            "SELECT * FROM users WHERE userEmail = $1", user.userEmail
        )

        responseMessage = None
        otp = generateOTP()

        if email and len(email) > 0:

            # all fields are autoconverted to lowercase in postgres
            accountStatus = email[0].get("accountStatus".lower())

            if accountStatus == "ACTIVE":
                raise HTTPException(
                    status_code=400, detail="User already registered and active"
                )

            elif accountStatus == "REGISTERED":

                responseMessage = (
                    "User already registered but not active, new otp send to mail."
                )

                await con.execute(
                    "UPDATE otp SET otp = $1 WHERE userId = $2",
                    otp,
                    email[0].get("userId".lower()),
                )

        else:

            id = uuid.uuid4()

            await con.execute(
                "INSERT INTO users (userId, userName, userEmail, password, accountStatus) VALUES ($1, $2, $3, $4, $5)",
                id,
                user.userName.strip(),
                user.userEmail.strip(),
                user.password.strip(),
                "REGISTERED",
            )

            await con.execute("INSERT INTO otp (userId, otp) VALUES ($1, $2)", id, otp)

        message = MailMessage(user.userEmail, "Registration successful", f"OTP is '{otp}'")

        mailer.sendMail(user.userEmail, message)

        print("register_user(): User registered successfully")

        return JSONResponse(
            status_code=200,
            content=jsonable_encoder(
                {
                    "message": (
                        responseMessage
                        if responseMessage is not None
                        else "User registered successfully. Check Email for OTP."
                    )
                }
            ),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await db._connection_pool.release(con)
